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

# Put the gate and all of its curl children in their own process group so a
# detected server crash can terminate the waiting request immediately.
setsid "$@" &
CHILD=$!

# The probe deliberately removes any stale container at startup.  Do not treat
# that cleanup transition as a crash.  Container-exit detection is armed only
# after this watchdog has observed the newly launched container in Running=true.
SEEN_RUNNING=0

kill_child_group() {
  kill -TERM -- "-$CHILD" >/dev/null 2>&1 || true
  for _ in 1 2 3 4 5; do
    kill -0 "$CHILD" >/dev/null 2>&1 || return 0
    sleep 0.1
  done
  kill -KILL -- "-$CHILD" >/dev/null 2>&1 || true
}

dump_failure_log() {
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
    running="$(docker inspect -f '{{.State.Running}}' "$CONTAINER" 2>/dev/null || echo false)"

    if [[ "$running" == true ]]; then
      SEEN_RUNNING=1
    fi

    # A container can briefly be marked for removal while startup cleanup is
    # racing with this watcher.  Capture logs only when Docker can actually
    # serve them; keep the last good copy otherwise.
    docker logs "$CONTAINER" >"$TMP/log.new" 2>&1 || true
    if [[ -s "$TMP/log.new" ]] && ! grep -q '^Error response from daemon:' "$TMP/log.new"; then
      mv "$TMP/log.new" "$TMP/log"
    else
      rm -f "$TMP/log.new"
    fi

    if [[ -s "$TMP/log" ]] && grep -Eq "$FATAL_RE" "$TMP/log"; then
      echo
      echo 'FAILFAST_SERVER_ERROR=True'
      echo 'Detected fatal scheduler/worker error; aborting the waiting request.'
      kill_child_group
      dump_failure_log
      exit 90
    fi

    if [[ "$SEEN_RUNNING" == 1 && "$running" != true ]]; then
      echo
      echo 'FAILFAST_CONTAINER_EXIT=True'
      echo 'Container exited after the probe had observed it running.'
      kill_child_group
      dump_failure_log
      exit 91
    fi
  fi

  sleep "$POLL_INTERVAL"
done

set +e
wait "$CHILD"
RC=$?
set -e
exit "$RC"
