#!/usr/bin/env bash
set -u

CONTAINER="${1:-sglang-qwen38-test}"

if ! docker inspect "$CONTAINER" >/dev/null 2>&1; then
  echo "container not found: $CONTAINER" >&2
  exit 1
fi

STARTED="$(docker inspect -f '{{.State.StartedAt}}' "$CONTAINER")"
echo "=== CONTAINER ==="
echo "StartedAt: $STARTED"
docker inspect -f 'running={{.State.Running}} status={{.State.Status}} exit={{.State.ExitCode}} oom={{.State.OOMKilled}} pid={{.State.Pid}}' "$CONTAINER" || true

echo "=== HTTP HEALTH (3s timeout) ==="
curl --max-time 3 -sS -w '\nhttp=%{http_code} total=%{time_total}s\n' http://127.0.0.1:30000/model_info || true

echo "=== PROCESS TREE ==="
docker top "$CONTAINER" -eo pid,ppid,stat,etime,pcpu,pmem,args 2>&1 | tail -80 || true

echo "=== NVIDIA-SMI ==="
nvidia-smi || true

echo "=== NVIDIA-SMI PMON ==="
nvidia-smi pmon -c 1 2>&1 || true

echo "=== CUTOVER / ERROR MARKERS ==="
docker logs --since "$STARTED" "$CONTAINER" 2>&1 | \
  grep -E 'MTP-CUTOVER|Prefill batch|Decode batch|Scheduler hit an exception|Traceback|Exception|RuntimeError|ValueError|AssertionError|AttributeError|CUDA error|illegal memory|out of memory|NCCL|watchdog|SIG|Killed' | \
  tail -700 || true

echo "=== LAST 500 LOG LINES ==="
docker logs --since "$STARTED" "$CONTAINER" 2>&1 | tail -500 || true

echo "=== END NON-DESTRUCTIVE HANG DUMP ==="
