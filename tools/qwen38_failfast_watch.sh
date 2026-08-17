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

kill_child_group() {
  kill -TERM -- "-$CHILD" >/dev/null 2>&1 || true
  for _ in 1 2 3 4 5; do
    kill -0 "$CHILD" >/dev/null 2>&1 || return 0
    sleep 0.1
  done
  kill -KILL -- "-$CHILD" >/dev/null 2>&1 || true
}

while kill -0 "$CHILD" >/dev/null 2>&1; do
  if docker inspect "$CONTAINER" >/dev/null 2>&1; then
    docker logs "$CONTAINER" >"$TMP/log" 2>&1 || true

    if grep -Eq "$FATAL_RE" "$TMP/log"; then
      echo
      echo 'FAILFAST_SERVER_ERROR=True'
      echo 'Detected fatal scheduler/worker error; aborting the waiting request.'
      kill_child_group
      echo '=== IMMEDIATE FAILURE LOG ==='
      grep -Ei 'PP[0-9]|MTP|speculative|Scheduler hit an exception|Traceback|AssertionError|RuntimeError|ValueError|CUDA error|out of memory|Exception triggered' "$TMP/log" | tail -220 || true
      exit 90
    fi

    running="$(docker inspect -f '{{.State.Running}}' "$CONTAINER" 2>/dev/null || echo false)"
    if [[ "$running" != true ]]; then
      echo
      echo 'FAILFAST_CONTAINER_EXIT=True'
      echo 'Container exited while the probe was still waiting.'
      kill_child_group
      echo '=== IMMEDIATE FAILURE LOG ==='
      tail -220 "$TMP/log" || true
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
