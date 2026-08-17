#!/usr/bin/env bash
set -euo pipefail

if (($# == 0)); then
  echo "usage: $0 command [args...]" >&2
  exit 64
fi

CONTAINER="${CONTAINER:-sglang-qwen38-gittensor-pp3}"
POLL_INTERVAL="${FAILFAST_POLL_INTERVAL:-0.5}"

# Deliberately exclude known informational PP/spec messages such as
# "Pipeline parallelism is incompatible with overlap schedule."  We only
# stop on evidence that a worker/scheduler actually crashed.
FATAL_RE='Scheduler hit an exception|Traceback \(most recent call last\)|AssertionError|RuntimeError:|ValueError:|CUDA error|out of memory|Exception triggered'

TMP="$(mktemp -d /tmp/qwen38-failfast.XXXXXX)"
trap 'rm -rf "$TMP"' EXIT

# The gate removes any stale container with this same name before launching a
# new one.  Record the old identity BEFORE starting the gate so a still-running
# stale container can never arm exit detection for the new probe.
BASE_CONTAINER_ID="$(docker inspect -f '{{.Id}}' "$CONTAINER" 2>/dev/null || true)"
ACTIVE_CONTAINER_ID=""
SEEN_RUNNING=0

# Put the gate and all of its curl children in their own process group so a
# detected server crash can terminate the waiting request immediately.
setsid "$@" &
CHILD=$!

kill_child_group() {
  kill -TERM -- "-$CHILD" >/dev/null 2>&1 || true
  for _ in 1 2 3 4 5; do
    kill -0 "$CHILD" >/dev/null 2>&1 || return 0
    sleep 0.1
  done
  kill -KILL -- "-$CHILD" >/dev/null 2>&1 || true
}

capture_logs() {
  docker logs "$CONTAINER" >"$TMP/log.new" 2>&1 || true
  if [[ -s "$TMP/log.new" ]] && ! grep -q '^Error response from daemon:' "$TMP/log.new"; then
    mv "$TMP/log.new" "$TMP/log"
  else
    rm -f "$TMP/log.new"
  fi
}

capture_state() {
  docker inspect -f \
    'id={{.Id}} name={{.Name}} status={{.State.Status}} running={{.State.Running}} exit_code={{.State.ExitCode}} oom_killed={{.State.OOMKilled}} error={{printf "%q" .State.Error}} started_at={{.State.StartedAt}} finished_at={{.State.FinishedAt}}' \
    "$CONTAINER" >"$TMP/state" 2>&1 || true
}

flush_exit_evidence() {
  # Docker can report Running=false a fraction of a second before the json-file
  # log driver has exposed the final stderr/stdout bytes.  Give it a short,
  # bounded grace period and keep the last non-error snapshot.
  capture_state
  for _ in 1 2 3 4 5 6 7 8; do
    capture_logs
    sleep 0.125
  done
  capture_state
}

dump_failure_log() {
  echo '=== IMMEDIATE CONTAINER STATE ==='
  if [[ -s "$TMP/state" ]]; then
    cat "$TMP/state"
  else
    echo '(container state unavailable)'
  fi

  echo '=== IMMEDIATE FAILURE SUMMARY ==='
  if [[ -s "$TMP/log" ]]; then
    grep -Ei 'PP[0-9]|MTP|speculative|Scheduler hit an exception|Traceback|AssertionError|RuntimeError|ValueError|CUDA error|out of memory|Exception triggered' "$TMP/log" | tail -80 || true
    echo
    echo '=== IMMEDIATE FAILURE RAW TAIL ==='
    # Do not grep the traceback here: Python continuation frames and the final
    # exception message often do not contain any of the summary keywords.
    tail -260 "$TMP/log" || true
  else
    echo '(no container log captured)'
  fi
}

while kill -0 "$CHILD" >/dev/null 2>&1; do
  if docker inspect "$CONTAINER" >/dev/null 2>&1; then
    current_id="$(docker inspect -f '{{.Id}}' "$CONTAINER" 2>/dev/null || true)"
    running="$(docker inspect -f '{{.State.Running}}' "$CONTAINER" 2>/dev/null || echo false)"

    # Arm only on a genuinely new container identity.  Merely seeing a running
    # container with the reused name is insufficient because startup cleanup
    # may still be removing the previous probe instance.
    if [[ -z "$ACTIVE_CONTAINER_ID" ]]; then
      if [[ -n "$current_id" && "$current_id" != "$BASE_CONTAINER_ID" && "$running" == true ]]; then
        ACTIVE_CONTAINER_ID="$current_id"
        SEEN_RUNNING=1
      fi
    fi

    # Once armed, ignore any later container that happens to reuse the same
    # name unless it is the exact container instance we observed starting.
    if [[ "$SEEN_RUNNING" == 1 && "$current_id" == "$ACTIVE_CONTAINER_ID" ]]; then
      capture_logs

      if [[ -s "$TMP/log" ]] && grep -Eq "$FATAL_RE" "$TMP/log"; then
        echo
        echo 'FAILFAST_SERVER_ERROR=True'
        echo 'Detected fatal scheduler/worker error; aborting the waiting request.'
        capture_state
        kill_child_group
        dump_failure_log
        exit 90
      fi

      if [[ "$running" != true ]]; then
        echo
        echo 'FAILFAST_CONTAINER_EXIT=True'
        echo 'The newly launched probe container exited after Running=true.'
        # Preserve evidence before terminating the gate: the gate itself also
        # notices container death and may otherwise race this watcher while it
        # is trying to print its own diagnostics.
        flush_exit_evidence
        kill_child_group
        dump_failure_log
        exit 91
      fi
    fi
  elif [[ "$SEEN_RUNNING" == 1 ]]; then
    # This path should be rare because the gate does not use --rm, but make a
    # removed active container explicit rather than silently losing evidence.
    printf 'id=%s status=removed_after_running\n' "$ACTIVE_CONTAINER_ID" >"$TMP/state"
    echo
    echo 'FAILFAST_CONTAINER_REMOVED=True'
    echo 'The newly launched probe container disappeared after Running=true.'
    kill_child_group
    dump_failure_log
    exit 92
  fi

  sleep "$POLL_INTERVAL"
done

set +e
wait "$CHILD"
RC=$?
set -e
exit "$RC"
