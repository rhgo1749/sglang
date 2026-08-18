#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-sglang:qwen38-27b-pp-mtp-share}"
MODEL="${MODEL:-gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090}"
CONTAINER="${CONTAINER:-sglang-qwen38-gittensor-pp2-single}"
PORT="${PORT:-30001}"
PARTITION="${PARTITION:-31,33}"
CTX="${CTX:-262144}"
MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-262144}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.84}"
CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-1024}"
ROOT="${ROOT:-/tmp/qwen38-pp2-dual5070-single}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-1800}"

cleanup() { docker rm -f "$CONTAINER" >/dev/null 2>&1 || true; }
trap cleanup EXIT INT TERM

dump_errors() {
  echo '--- FIRST/ROOT SERVER ERROR CONTEXT ---'
  docker logs "$CONTAINER" 2>&1 | grep -Ei -m 1 -B 20 -A 80 'Traceback|AssertionError|RuntimeError|ValueError|CUDA out of memory|out of memory|Exception' || \
    docker logs "$CONTAINER" 2>&1 | tail -240 || true
}

if (( MAX_TOTAL_TOKENS < CTX )); then
  echo "ERROR: max_total_tokens=${MAX_TOTAL_TOKENS} < context=${CTX}" >&2
  exit 64
fi

mkdir -p "$ROOT"
cleanup
sleep 2

echo '============================================================'
echo ' Qwen3.8 FINAL PP2 / DUAL-5070 / SINGLE-REQUEST GATE'
echo " image=${IMAGE}"
echo " model=${MODEL}"
echo " physical_gpus=0,2 (RTX 5070 Ti x2)"
echo " pp=2 partition=${PARTITION}"
echo " context=${CTX} pool=${MAX_TOTAL_TOKENS} max_running=1 mamba_slots=1"
echo " chunked_prefill=${CHUNKED_PREFILL_SIZE}"
echo ' native MTP: steps=3 topk=1 draft_tokens=4'
echo ' CUDA graph: OFF'
echo '============================================================'

docker run -d \
  --name "$CONTAINER" \
  --gpus '"device=0,2"' \
  --ipc=host \
  --shm-size 32g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -e "SGLANG_PP_LAYER_PARTITION=${PARTITION}" \
  -e 'PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True' \
  -p "${PORT}:30000" \
  -v sglang-hf-cache:/root/.cache/huggingface \
  "$IMAGE" \
  python3 -m sglang.launch_server \
    --model-path "$MODEL" \
    --tp-size 1 \
    --pp-size 2 \
    --trust-remote-code \
    --context-length "$CTX" \
    --max-total-tokens "$MAX_TOTAL_TOKENS" \
    --speculative-algorithm NEXTN \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 4 \
    --skip-server-warmup \
    --kv-cache-dtype nvfp4 \
    --attention-backend flashinfer \
    --prefill-attention-backend flashinfer \
    --decode-attention-backend trtllm_mha \
    --page-size 64 \
    --mem-fraction-static "$MEM_FRACTION_STATIC" \
    --max-running-requests 1 \
    --max-mamba-cache-size 1 \
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
  [[ "$running" == true ]] || exit 2
  sleep 1
done
'; then
  echo 'BOOT_PASS=False'
  dump_errors
  exit 1
fi

echo 'BOOT_PASS=True'
LOG="$ROOT/boot.log"
docker logs "$CONTAINER" >"$LOG" 2>&1 || true
CAP="$(python3 - "$LOG" <<'PY'
import re,sys
text=open(sys.argv[1],errors='replace').read()
vals=[int(x) for x in re.findall(r'max_total_num_tokens[^0-9]*([0-9]+)',text,re.I)]
print(min(vals) if vals else 0)
PY
)"
echo "final_capacity=${CAP}"
echo "required=${CTX}"
if (( CAP < CTX )); then
  echo 'FINAL_TARGET_CAPACITY=FAIL'
  docker logs "$CONTAINER" 2>&1 | grep -E 'MTP-PP-(LOCAL-MEM-BUDGET|CAPACITY-LOCAL|CAPACITY-GLOBAL)|max_total_tokens=.*profiled' || true
  exit 2
fi
echo 'FINAL_TARGET_CAPACITY=PASS'

# Semantic sanity check before timing.
cat >"$ROOT/semantic.json" <<JSON
{"model":"${MODEL}","messages":[{"role":"user","content":"Reply with exactly: TEXT_CONTROL_OK"}],"temperature":0,"max_tokens":256}
JSON
HTTP="$(curl --max-time 180 -sS -o "$ROOT/semantic-response.json" -w '%{http_code}' \
  "http://127.0.0.1:${PORT}/v1/chat/completions" -H 'Content-Type: application/json' --data-binary "@$ROOT/semantic.json")"
python3 - "$ROOT/semantic-response.json" "$HTTP" <<'PY'
import json,sys
p,http=sys.argv[1:]
d=json.load(open(p)); m=(d.get('choices') or [{}])[0].get('message') or {}
text=' '.join(str(m.get(k) or '') for k in ('reasoning_content','content'))
ok=http=='200' and 'TEXT_CONTROL_OK' in text
print(f'SEMANTIC_PASS={ok}')
raise SystemExit(0 if ok else 1)
PY

run_case() {
  local label="$1" prompt_tokens="$2" decode_tokens="$3"
  python3 - "$ROOT/${label}-request.json" "$prompt_tokens" "$decode_tokens" <<'PY'
import json,sys
path,p,d=sys.argv[1],int(sys.argv[2]),int(sys.argv[3])
json.dump({'input_ids':[1200]*p,'sampling_params':{'temperature':0,'max_new_tokens':d,'ignore_eos':True}},open(path,'w'),separators=(',',':'))
PY
  local start end http rc
  start="$(date +%s%N)"
  set +e
  http="$(curl --max-time "$REQUEST_TIMEOUT" -sS -o "$ROOT/${label}-response.json" -w '%{http_code}' \
    "http://127.0.0.1:${PORT}/generate" -H 'Content-Type: application/json' --data-binary "@$ROOT/${label}-request.json")"
  rc=$?
  set -e
  end="$(date +%s%N)"
  python3 - "$ROOT/${label}-response.json" "$label" "$prompt_tokens" "$decode_tokens" "$http" "$rc" "$start" "$end" <<'PY'
import json,sys
path,label,p,d,http,rc,start,end=sys.argv[1],sys.argv[2],int(sys.argv[3]),int(sys.argv[4]),sys.argv[5],int(sys.argv[6]),int(sys.argv[7]),int(sys.argv[8])
data=json.load(open(path)); m=data.get('meta_info') or {}
pn=int(m.get('prompt_tokens') or 0); cn=int(m.get('completion_tokens') or 0)
elapsed=(end-start)/1e9
ok=rc==0 and http=='200' and pn==p and cn==d
print(f'{label}_http={http}')
print(f'{label}_wall_seconds={elapsed:.3f}')
print(f'{label}_prompt_tokens={pn}')
print(f'{label}_completion_tokens={cn}')
for key in ('prompt_tokens','completion_tokens','completion_tokens_wo_jump_forward'):
    if key in m: print(f'{label}_meta_{key}={m[key]}')
print(f'{label}_exact_token_pass={ok}')
raise SystemExit(0 if ok else 1)
PY
}

echo '=== SHORT 4096+1024 / SINGLE ==='
run_case short 4096 1024

echo '=== LONG 262000+8 / SINGLE ==='
run_case long 262000 8

echo '=== FINAL PP2 SUMMARY ==='
echo 'llamacpp_short_steady_reference_seconds=6.692'
echo 'llamacpp_long_mean_reference_seconds=201.207'
echo 'QWEN38_GITTENSOR_PP2_DUAL5070_SINGLE_GATE=PASS'
