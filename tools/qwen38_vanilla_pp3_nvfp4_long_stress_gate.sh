#!/usr/bin/env bash
set -euo pipefail

# Long-context follow-up for qwen38_vanilla_pp3_nvfp4_final_gate.sh.
# Assumes the validated server is already running with:
#   PP=19,23,22 / NVFP4 KV / 262144 ctx / max_running_requests=3
# This script does not restart or mutate the server.

PORT="${PORT:-30000}"
CONTAINER="${CONTAINER:-sglang-qwen38-vanilla-pp3-final}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-262144}"
DECODE_TOKENS="${DECODE_TOKENS:-8}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-1200}"
STAGES="${STAGES:-65536 131072 196608 258048}"
PARALLEL="${PARALLEL:-3}"
ROOT="${ROOT:-/tmp/qwen38-long-stress}"
MONITOR_PID=""

cleanup_monitor() {
  if [[ -n "${MONITOR_PID:-}" ]]; then
    kill "$MONITOR_PID" >/dev/null 2>&1 || true
    wait "$MONITOR_PID" 2>/dev/null || true
    MONITOR_PID=""
  fi
}
trap cleanup_monitor EXIT INT TERM

if (( PARALLEL != 3 )); then
  echo "ERROR: this gate is intentionally fixed to PARALLEL=3"
  exit 64
fi

if ! curl -fsS --max-time 5 "http://127.0.0.1:${PORT}/model_info" >/dev/null; then
  echo "ERROR: no healthy SGLang server detected on port ${PORT}"
  echo "Run tools/qwen38_vanilla_pp3_nvfp4_final_gate.sh first and leave the container running."
  exit 65
fi

echo '============================================================'
echo ' QWEN3.8 VANILLA PP3 + NVFP4 REAL LONG-CONTEXT STRESS GATE'
echo " port=${PORT}"
echo " container=${CONTAINER}"
echo " context_length=${CONTEXT_LENGTH}"
echo " stages=${STAGES}"
echo " parallel=${PARALLEL}"
echo " decode_tokens=${DECODE_TOKENS}"
echo " request_timeout=${REQUEST_TIMEOUT}s"
echo ' expected baseline: PP=19,23,22 / 3x262144 capacity already PASS'
echo '============================================================'

rm -rf "$ROOT"
mkdir -p "$ROOT"

monitor_gpu() {
  local out="$1"
  : >"$out"
  while true; do
    date +%s%3N | tr -d '\n' >>"$out"
    printf ' ' >>"$out"
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
      --format=csv,noheader,nounits | tr '\n' ';' >>"$out" || true
    printf '\n' >>"$out"
    sleep 1
  done
}

summarize_gpu() {
  local file="$1"
  python3 - "$file" <<'PY'
import re,sys
p=sys.argv[1]
peak={}
try:
    lines=open(p,errors='replace')
except OSError:
    lines=[]
for line in lines:
    # rows are serialized as: idx, used, total, util;idx, used, total, util;
    for m in re.finditer(r'(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)', line):
        idx,used,total,util=map(int,m.groups())
        cur=peak.get(idx,(0,total,0))
        peak[idx]=(max(cur[0],used),total,max(cur[2],util))
for idx in sorted(peak):
    used,total,util=peak[idx]
    print(f'gpu{idx}_peak_used_mib={used}')
    print(f'gpu{idx}_peak_free_mib={total-used}')
    print(f'gpu{idx}_peak_util_pct={util}')
PY
}

dump_server_errors() {
  echo '--- SERVER ERROR TAIL ---'
  docker logs "$CONTAINER" 2>&1 | \
    grep -Ei 'PP[0-9]|Traceback|RuntimeError|CUDA|out of memory|exception|watchdog' | \
    tail -300 || true
}

run_stage() {
  local tokens="$1"
  local stage_dir="$ROOT/$tokens"
  mkdir -p "$stage_dir"

  echo
  echo "=== REAL PARALLEL-3 STAGE: prompt_tokens=${tokens} decode_tokens=${DECODE_TOKENS} ==="

  for i in 0 1 2; do
    python3 - "$stage_dir/request-${i}.json" "$tokens" "$DECODE_TOKENS" "$i" <<'PY'
import json,sys
path,tokens,decode,i=sys.argv[1],int(sys.argv[2]),int(sys.argv[3]),int(sys.argv[4])
# Different token IDs prevent the three sequences from being byte-identical.
payload={
    'input_ids':[1000+i]*tokens,
    'sampling_params':{
        'temperature':0,
        'max_new_tokens':decode,
        'ignore_eos':True,
    },
}
with open(path,'w') as f:
    json.dump(payload,f,separators=(',',':'))
PY
  done

  monitor_gpu "$stage_dir/gpu.log" &
  MONITOR_PID=$!

  local start_ns
  start_ns="$(date +%s%N)"
  local pids=()
  for i in 0 1 2; do
    (
      set +e
      http="$(curl --max-time "$REQUEST_TIMEOUT" -sS \
        -o "$stage_dir/response-${i}.json" \
        -w '%{http_code}' \
        "http://127.0.0.1:${PORT}/generate" \
        -H 'Content-Type: application/json' \
        --data-binary "@$stage_dir/request-${i}.json")"
      rc=$?
      printf '%s\n' "$http" >"$stage_dir/http-${i}.txt"
      printf '%s\n' "$rc" >"$stage_dir/rc-${i}.txt"
      exit 0
    ) &
    pids+=("$!")
  done

  for pid in "${pids[@]}"; do
    wait "$pid" || true
  done
  local end_ns
  end_ns="$(date +%s%N)"

  cleanup_monitor

  set +e
  python3 - "$stage_dir" "$tokens" "$DECODE_TOKENS" "$start_ns" "$end_ns" <<'PY'
import json,pathlib,sys
root=pathlib.Path(sys.argv[1])
tokens=int(sys.argv[2]); decode=int(sys.argv[3])
start=int(sys.argv[4]); end=int(sys.argv[5])
ok=True
for i in range(3):
    def read(name,default=''):
        p=root/name
        return p.read_text().strip() if p.exists() else default
    http=read(f'http-{i}.txt')
    rc=read(f'rc-{i}.txt','999')
    print(f'req{i}_curl_rc={rc}')
    print(f'req{i}_http={http}')
    try:
        d=json.load(open(root/f'response-{i}.json'))
        m=d.get('meta_info',{})
        p=int(m.get('prompt_tokens') or 0)
        c=int(m.get('completion_tokens') or 0)
        print(f'req{i}_prompt_tokens={p}')
        print(f'req{i}_completion_tokens={c}')
        ok &= rc=='0' and http=='200' and p==tokens and c==decode
    except Exception as e:
        print(f'req{i}_parse_error={e!r}')
        ok=False
elapsed=(end-start)/1e9
print(f'stage_elapsed_seconds={elapsed:.3f}')
print(f'stage_parallel3_pass={ok}')
raise SystemExit(0 if ok else 1)
PY
  local verify_rc=$?
  set -e

  summarize_gpu "$stage_dir/gpu.log"

  echo '--- GPU STATE AFTER STAGE ---'
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,pstate \
    --format=csv || true

  if (( verify_rc != 0 )); then
    echo "LONG_STRESS_STAGE_${tokens}=FAIL"
    dump_server_errors
    return 1
  fi

  if ! curl -fsS --max-time 5 "http://127.0.0.1:${PORT}/model_info" >/dev/null; then
    echo "LONG_STRESS_STAGE_${tokens}=FAIL_SERVER_UNHEALTHY"
    dump_server_errors
    return 1
  fi

  echo "LONG_STRESS_STAGE_${tokens}=PASS"
}

for tokens in $STAGES; do
  # Keep room for decode inside the model context window.
  if (( tokens + DECODE_TOKENS > CONTEXT_LENGTH )); then
    echo "ERROR: stage ${tokens}+${DECODE_TOKENS} exceeds ${CONTEXT_LENGTH} context"
    exit 66
  fi
  if ! run_stage "$tokens"; then
    echo
    echo 'QWEN38_PP3_NVFP4_REAL_LONG_STRESS_GATE=FAIL'
    exit 1
  fi
done

echo
echo '============================================================'
echo ' REAL 3-SESSION LONG-CONTEXT STRESS RESULT: PASS'
echo ' 3x64K -> 3x128K -> 3x192K -> 3x258048 all completed'
echo '============================================================'
echo 'QWEN38_PP3_NVFP4_REAL_LONG_STRESS_GATE=PASS'
