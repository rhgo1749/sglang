#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-sglang:qwen38-27b-pp-mtp-share}"
MODEL="${MODEL:-gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090}"
CONTAINER="${CONTAINER:-sglang-qwen38-gittensor-pp3}"
PORT="${PORT:-30001}"

CTX="${CTX:-262144}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-262144}"
PARALLEL="${PARALLEL:-2}"
MAX_RUNNING="${MAX_RUNNING:-2}"
MAX_MAMBA="${MAX_MAMBA:-2}"
PARTITION="${PARTITION:-22,28,14}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.84}"
MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-524288}"
CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-512}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

ENABLE_MTP="${ENABLE_MTP:-1}"
SPECULATIVE_ALGO="${SPECULATIVE_ALGO:-NEXTN}"
SPECULATIVE_NUM_STEPS="${SPECULATIVE_NUM_STEPS:-3}"
SPECULATIVE_EAGLE_TOPK="${SPECULATIVE_EAGLE_TOPK:-1}"
SPECULATIVE_NUM_DRAFT_TOKENS="${SPECULATIVE_NUM_DRAFT_TOKENS:-4}"

STAGE="${STAGE:-262000}"
DECODE_TOKENS="${DECODE_TOKENS:-8}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-1200}"
REQUIRED=$((CTX * MAX_RUNNING))
ROOT="${ROOT:-/tmp/qwen38-x2-256k}"
MONITOR_PID=""

if (( PARALLEL != 2 || MAX_RUNNING != 2 || MAX_MAMBA != 2 )); then
  echo "ERROR: this gate is intentionally fixed to PARALLEL=MAX_RUNNING=MAX_MAMBA=2"
  exit 64
fi
if (( MAX_TOTAL_TOKENS < REQUIRED )); then
  echo "ERROR: max_total_tokens=${MAX_TOTAL_TOKENS} < required=${REQUIRED}"
  exit 65
fi
if (( STAGE + DECODE_TOKENS > CONTEXT_LENGTH )); then
  echo "ERROR: stage ${STAGE}+${DECODE_TOKENS} exceeds context ${CONTEXT_LENGTH}"
  exit 66
fi

MTP_ARGS=()
if [[ "$ENABLE_MTP" == "1" ]]; then
  MTP_ARGS=(
    --speculative-algorithm "$SPECULATIVE_ALGO"
    --speculative-num-steps "$SPECULATIVE_NUM_STEPS"
    --speculative-eagle-topk "$SPECULATIVE_EAGLE_TOPK"
    --speculative-num-draft-tokens "$SPECULATIVE_NUM_DRAFT_TOKENS"
    --skip-server-warmup
  )
fi

cleanup_monitor() {
  if [[ -n "${MONITOR_PID:-}" ]]; then
    kill "$MONITOR_PID" >/dev/null 2>&1 || true
    wait "$MONITOR_PID" 2>/dev/null || true
    MONITOR_PID=""
  fi
}
trap cleanup_monitor EXIT INT TERM

cleanup_container() {
  docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
}

dump_errors() {
  echo '--- SERVER ERROR TAIL ---'
  docker logs "$CONTAINER" 2>&1 | \
    grep -Ei 'PP[0-9]|Traceback|AssertionError|RuntimeError|ValueError|CUDA|out of memory|exception|watchdog' | \
    tail -400 || true
}

monitor_gpu() {
  local out="$1"
  : >"$out"
  while true; do
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
      --format=csv,noheader,nounits | tr '\n' ';' >>"$out" || true
    printf '\n' >>"$out"
    sleep 1
  done
}

summarize_gpu() {
  python3 - "$1" <<'PY'
import re,sys
peak={}
try:
    lines=open(sys.argv[1], errors='replace')
except OSError:
    lines=[]
for line in lines:
    for m in re.finditer(r'(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)', line):
        idx,used,total,util=map(int,m.groups())
        old=peak.get(idx,(0,total,0))
        peak[idx]=(max(old[0],used),total,max(old[2],util))
for idx in sorted(peak):
    used,total,util=peak[idx]
    print(f'gpu{idx}_peak_used_mib={used}')
    print(f'gpu{idx}_peak_free_mib={total-used}')
    print(f'gpu{idx}_peak_util_pct={util}')
PY
}

echo '============================================================'
echo ' QWEN3.8 NATIVE MTP + PP3 + NVFP4 / 2x256K GATE'
echo " model=${MODEL}"
echo " context_length=${CTX}"
echo " parallel=${PARALLEL}"
echo " required_capacity=${REQUIRED}"
echo " partition=${PARTITION}"
echo " mem_fraction_static=${MEM_FRACTION_STATIC}"
echo " max_total_tokens=${MAX_TOTAL_TOKENS}"
echo " chunked_prefill_size=${CHUNKED_PREFILL_SIZE}"
echo " mamba_slots=${MAX_MAMBA}"
echo " long_prompt_tokens=${STAGE}"
echo " decode_tokens=${DECODE_TOKENS}"
echo ' MTP ON / CUDA GRAPH OFF / SERVER WARMUP OFF'
echo '============================================================'

cleanup_container
sleep 2

docker run -d \
  --name "$CONTAINER" \
  --gpus '"device=0,2,1"' \
  --ipc=host \
  --shm-size 32g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -e "SGLANG_PP_LAYER_PARTITION=${PARTITION}" \
  -e "PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF}" \
  -p "${PORT}:30000" \
  -v sglang-hf-cache:/root/.cache/huggingface \
  "$IMAGE" \
  python3 -m sglang.launch_server \
    --model-path "$MODEL" \
    --tp-size 1 \
    --pp-size 3 \
    --trust-remote-code \
    --context-length "$CTX" \
    --max-total-tokens "$MAX_TOTAL_TOKENS" \
    "${MTP_ARGS[@]}" \
    --kv-cache-dtype nvfp4 \
    --attention-backend flashinfer \
    --prefill-attention-backend flashinfer \
    --decode-attention-backend trtllm_mha \
    --page-size 64 \
    --mem-fraction-static "$MEM_FRACTION_STATIC" \
    --max-running-requests "$MAX_RUNNING" \
    --max-mamba-cache-size "$MAX_MAMBA" \
    --mamba-ssm-dtype bfloat16 \
    --mamba-radix-cache-strategy extra_buffer_lazy \
    --disable-radix-cache \
    --mm-feature-transport cpu \
    --chunked-prefill-size "$CHUNKED_PREFILL_SIZE" \
    --disable-cuda-graph \
    --disable-flashinfer-autotune \
    --reasoning-parser qwen3 \
    --tool-call-parser qwen3_coder \
    --host 0.0.0.0 \
    --port 30000 >/dev/null

if ! timeout 300 bash -c '
while true; do
  if curl -fsS http://127.0.0.1:'"$PORT"'/model_info >/dev/null 2>&1; then exit 0; fi
  running="$(docker inspect -f "{{.State.Running}}" '"$CONTAINER"' 2>/dev/null || echo false)"
  if [[ "$running" != true ]]; then exit 2; fi
  sleep 1
done
'; then
  echo 'BOOT_PASS=False'
  dump_errors
  exit 1
fi

echo 'BOOT_PASS=True'
LOG="${ROOT}-boot.log"
docker logs "$CONTAINER" >"$LOG" 2>&1 || true
CAP="$(python3 - "$LOG" <<'PY'
import re,sys
text=open(sys.argv[1],errors='replace').read()
vals=[int(x) for x in re.findall(r'max_total_num_tokens[^0-9]*([0-9]+)', text, re.I)]
print(min(vals) if vals else 0)
PY
)"
echo "final_capacity=${CAP}"
echo "required=${REQUIRED}"
if (( CAP < REQUIRED )); then
  echo 'FINAL_TARGET_CAPACITY=FAIL'
  dump_errors
  exit 2
fi
echo 'FINAL_TARGET_CAPACITY=PASS'

python3 - <<'PY'
import json
json.dump({'input_ids':[1000]*256,'sampling_params':{'temperature':0,'max_new_tokens':8,'ignore_eos':True}},open('/tmp/qwen38-x2-single.json','w'))
PY
HTTP="$(curl --max-time 60 -sS -o /tmp/qwen38-x2-single-response.json -w '%{http_code}' \
  "http://127.0.0.1:${PORT}/generate" -H 'Content-Type: application/json' \
  --data-binary @/tmp/qwen38-x2-single.json)"
python3 - "$HTTP" <<'PY'
import json,sys
http=sys.argv[1]
d=json.load(open('/tmp/qwen38-x2-single-response.json'))
m=d.get('meta_info',{})
ok=http=='200' and int(m.get('prompt_tokens') or 0)==256 and int(m.get('completion_tokens') or 0)==8
print(f'single_http={http}')
print(f'single_prompt_tokens={m.get("prompt_tokens")}')
print(f'single_completion_tokens={m.get("completion_tokens")}')
print(f'single_functional_pass={ok}')
raise SystemExit(0 if ok else 1)
PY

echo '=== PARALLEL-2 SHORT MAMBA SLOT CHECK ==='
for i in 0 1; do
  python3 - "$i" <<'PY'
import json,sys
i=int(sys.argv[1])
json.dump({'input_ids':[1100+i]*256,'sampling_params':{'temperature':0,'max_new_tokens':8,'ignore_eos':True}},open(f'/tmp/qwen38-x2-short-{i}.json','w'))
PY
done
PIDS=()
for i in 0 1; do
  (curl --max-time 90 -sS -o "/tmp/qwen38-x2-short-${i}-response.json" -w '%{http_code}' \
    "http://127.0.0.1:${PORT}/generate" -H 'Content-Type: application/json' \
    --data-binary "@/tmp/qwen38-x2-short-${i}.json" >"/tmp/qwen38-x2-short-${i}-http.txt") &
  PIDS+=("$!")
done
for pid in "${PIDS[@]}"; do wait "$pid"; done
python3 - <<'PY'
import json,pathlib
ok=True
for i in range(2):
    http=pathlib.Path(f'/tmp/qwen38-x2-short-{i}-http.txt').read_text().strip()
    d=json.load(open(f'/tmp/qwen38-x2-short-{i}-response.json'))
    m=d.get('meta_info',{})
    p=int(m.get('prompt_tokens') or 0); c=int(m.get('completion_tokens') or 0)
    print(f'req{i}_http={http}')
    print(f'req{i}_prompt_tokens={p}')
    print(f'req{i}_completion_tokens={c}')
    ok &= http=='200' and p==256 and c==8
print(f'parallel2_mamba_slot_pass={ok}')
raise SystemExit(0 if ok else 1)
PY

echo '=== REAL PARALLEL-2 256K-CONTEXT STRESS ==='
rm -rf "$ROOT"
mkdir -p "$ROOT"
for i in 0 1; do
  python3 - "$ROOT/request-${i}.json" "$STAGE" "$DECODE_TOKENS" "$i" <<'PY'
import json,sys
path,tokens,decode,i=sys.argv[1],int(sys.argv[2]),int(sys.argv[3]),int(sys.argv[4])
json.dump({'input_ids':[1200+i]*tokens,'sampling_params':{'temperature':0,'max_new_tokens':decode,'ignore_eos':True}},open(path,'w'),separators=(',',':'))
PY
done
monitor_gpu "$ROOT/gpu.log" &
MONITOR_PID=$!
START_NS="$(date +%s%N)"
PIDS=()
for i in 0 1; do
  (
    set +e
    http="$(curl --max-time "$REQUEST_TIMEOUT" -sS -o "$ROOT/response-${i}.json" -w '%{http_code}' \
      "http://127.0.0.1:${PORT}/generate" -H 'Content-Type: application/json' \
      --data-binary "@$ROOT/request-${i}.json")"
    rc=$?
    printf '%s\n' "$http" >"$ROOT/http-${i}.txt"
    printf '%s\n' "$rc" >"$ROOT/rc-${i}.txt"
  ) &
  PIDS+=("$!")
done
for pid in "${PIDS[@]}"; do wait "$pid" || true; done
END_NS="$(date +%s%N)"
cleanup_monitor

set +e
python3 - "$ROOT" "$STAGE" "$DECODE_TOKENS" "$START_NS" "$END_NS" <<'PY'
import json,pathlib,sys
root=pathlib.Path(sys.argv[1]); tokens=int(sys.argv[2]); decode=int(sys.argv[3]); start=int(sys.argv[4]); end=int(sys.argv[5])
ok=True
for i in range(2):
    def read(name,default=''):
        p=root/name
        return p.read_text().strip() if p.exists() else default
    http=read(f'http-{i}.txt'); rc=read(f'rc-{i}.txt','999')
    print(f'req{i}_curl_rc={rc}')
    print(f'req{i}_http={http}')
    try:
        d=json.load(open(root/f'response-{i}.json')); m=d.get('meta_info',{})
        p=int(m.get('prompt_tokens') or 0); c=int(m.get('completion_tokens') or 0)
        print(f'req{i}_prompt_tokens={p}')
        print(f'req{i}_completion_tokens={c}')
        ok &= rc=='0' and http=='200' and p==tokens and c==decode
    except Exception as e:
        print(f'req{i}_parse_error={e!r}'); ok=False
print(f'stage_elapsed_seconds={(end-start)/1e9:.3f}')
print(f'stage_parallel2_pass={ok}')
raise SystemExit(0 if ok else 1)
PY
VERIFY_RC=$?
set -e
summarize_gpu "$ROOT/gpu.log"

echo '--- GPU STATE AFTER STAGE ---'
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,pstate --format=csv || true
if (( VERIFY_RC != 0 )); then
  echo 'X2_256K_LONG_STRESS=FAIL'
  dump_errors
  exit 3
fi
if ! curl -fsS --max-time 5 "http://127.0.0.1:${PORT}/model_info" >/dev/null; then
  echo 'X2_256K_LONG_STRESS=FAIL_SERVER_UNHEALTHY'
  dump_errors
  exit 4
fi

echo 'X2_256K_LONG_STRESS=PASS'
echo 'QWEN38_GITTENSOR_PP3_NVFP4_X2_256K_GATE=PASS'
