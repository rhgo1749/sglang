#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-lmsysorg/sglang:qwen38-27b}"
MODEL="${MODEL:-RadixArk/Qwen3.8-27B-NVFP4}"
CONTAINER="${CONTAINER:-sglang-qwen38-vanilla-pp3-nvfp4-capacity}"
PORT="${PORT:-30000}"
CTX="${CTX:-262144}"
MAX_RUNNING="${MAX_RUNNING:-3}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.95}"
REQUIRED=$((CTX * MAX_RUNNING))

cleanup_old() {
  docker rm -f \
    sglang-qwen38-test \
    sglang-qwen38-vanilla-pp3 \
    sglang-qwen38-vanilla-pp3-capacity \
    "$CONTAINER" >/dev/null 2>&1 || true
}

dump_logs() {
  echo '=== IMPORTANT LOGS ==='
  docker logs "$CONTAINER" 2>&1 | \
    grep -Ei 'pipeline|pp.rank|NVFP4|FP4|KV Cache|#tokens|max_total_num_tokens|Memory pool|available_gpu_mem|Mamba|Traceback|AssertionError|RuntimeError|ValueError|CUDA error|out of memory|NCCL|exception' | \
    tail -700 || true
  echo '=== LOG TAIL ==='
  docker logs "$CONTAINER" 2>&1 | tail -300 || true
  echo '=== GPU STATE ==='
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,pstate --format=csv || true
}

echo '============================================================'
echo ' QWEN3.8 PURE VANILLA PP3 + NVFP4 KV CAPACITY SMOKE — MTP OFF'
echo ' Goal: test whether three raw 262144-token sessions fit across PP3'
echo " Required per-stage token capacity: ${REQUIRED}"
echo " mem_fraction_static: ${MEM_FRACTION_STATIC}"
echo ' NO SOURCE PATCH / NO SOURCE MOUNT'
echo '============================================================'

echo '=== EXACT IMAGE CLI PREFLIGHT ==='
HELP="$(mktemp)"
docker run --rm "$IMAGE" python3 -m sglang.launch_server --help >"$HELP" 2>&1
for arg in '--pp-size' '--kv-cache-dtype' '--prefill-attention-backend' '--decode-attention-backend' '--page-size'; do
  grep -q -- "$arg" "$HELP" || { echo "ERROR: exact image lacks $arg"; exit 1; }
done
grep -q 'nvfp4' "$HELP" || { echo 'ERROR: exact image does not advertise nvfp4 KV'; exit 1; }
grep -q 'trtllm_mha' "$HELP" || { echo 'ERROR: exact image does not advertise trtllm_mha'; exit 1; }

EXTRA_ARGS=(--disable-cuda-graph)
if grep -q -- '--disable-flashinfer-autotune' "$HELP"; then
  EXTRA_ARGS+=(--disable-flashinfer-autotune)
  echo 'capacity mode: CUDA graph + FlashInfer autotune disabled'
else
  echo 'capacity mode: CUDA graph disabled'
fi

cleanup_old
sleep 2

echo '=== GPU STATE BEFORE LAUNCH ==='
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,pstate --format=csv

echo '=== START PURE VANILLA PP3 + NVFP4 KV, MTP OFF ==='
docker run -d \
  --name "$CONTAINER" \
  --gpus '"device=0,2,1"' \
  --ipc=host \
  --shm-size 32g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -p "${PORT}:30000" \
  -v sglang-hf-cache:/root/.cache/huggingface \
  "$IMAGE" \
  python3 -m sglang.launch_server \
    --model-path "$MODEL" \
    --tp-size 1 \
    --pp-size 3 \
    --trust-remote-code \
    --context-length "$CTX" \
    --kv-cache-dtype nvfp4 \
    --attention-backend flashinfer \
    --prefill-attention-backend flashinfer \
    --decode-attention-backend trtllm_mha \
    --page-size 64 \
    --mem-fraction-static "$MEM_FRACTION_STATIC" \
    --max-running-requests "$MAX_RUNNING" \
    --max-mamba-cache-size 8 \
    --mamba-ssm-dtype bfloat16 \
    --mamba-radix-cache-strategy extra_buffer_lazy \
    --disable-radix-cache \
    --mm-feature-transport cpu \
    --chunked-prefill-size 2048 \
    --reasoning-parser qwen3 \
    --tool-call-parser qwen3_coder \
    "${EXTRA_ARGS[@]}" \
    --host 0.0.0.0 \
    --port 30000 >/dev/null

STARTED="$(docker inspect -f '{{.State.StartedAt}}' "$CONTAINER")"

echo '=== VERIFY: NO SOURCE MOUNTS ==='
docker inspect "$CONTAINER" --format '{{range .Mounts}}{{println .Destination " <- " .Source}}{{end}}'
echo 'Only /root/.cache/huggingface should appear above.'

echo '=== WAIT FOR SERVER (max 300s) ==='
if ! timeout 300 bash -c '
while true; do
  if curl -fsS http://127.0.0.1:'"$PORT"'/model_info >/dev/null 2>&1; then exit 0; fi
  running="$(docker inspect -f "{{.State.Running}}" '"$CONTAINER"' 2>/dev/null || echo false)"
  if [[ "$running" != true ]]; then exit 2; fi
  sleep 1
done
'; then
  echo '❌ VANILLA PP3 + NVFP4 (MTP OFF) DID NOT BOOT'
  dump_logs
  exit 1
fi

echo '✅ VANILLA PP3 + NVFP4 (MTP OFF) BOOTED'
LOG="$(mktemp)"
docker logs --since "$STARTED" "$CONTAINER" >"$LOG" 2>&1 || true

echo '=== CAPACITY / TOPOLOGY ==='
grep -Ei 'pipeline|pp.rank|NVFP4|FP4|KV Cache|#tokens|max_total_num_tokens|Memory pool end|available_gpu_mem|Mamba Cache' "$LOG" | tail -500 || true

echo '=== TOKEN CAPACITY SUMMARY ==='
python3 - "$LOG" "$REQUIRED" <<'PY'
import re, sys
p, required = sys.argv[1], int(sys.argv[2])
text = open(p, errors='replace').read()
vals=[]
for pat in [r'max_total_num_tokens[^0-9]*([0-9]+)', r'#tokens:\s*([0-9]+)']:
    vals += [int(x) for x in re.findall(pat, text, flags=re.I)]
uniq=[]
for v in vals:
    if v not in uniq:
        uniq.append(v)
print('parsed_token_capacities=' + (','.join(map(str,uniq)) if uniq else 'NONE'))
if uniq:
    mn=min(uniq); mx=max(uniq)
    print(f'min_reported_capacity={mn}')
    print(f'max_reported_capacity={mx}')
    print(f'required_for_3x256k={required}')
    print(f'capacity_ratio_to_3x256k={mn/required:.4f}')
    print(f'raw_3x256k_capacity_gate={mn >= required}')
    print(f'equal_share_tokens_per_session={mn//3}')
else:
    print(f'required_for_3x256k={required}')
    print('raw_3x256k_capacity_gate=UNKNOWN')
PY

echo '=== CHEAP FUNCTIONAL REQUEST ==='
python3 - <<'PY'
import json
json.dump({
  'input_ids':[1000]*256,
  'sampling_params':{'temperature':0,'max_new_tokens':16,'ignore_eos':True},
}, open('/tmp/qwen38-vanilla-pp3-nvfp4-one.json','w'))
PY
HTTP="$(curl --max-time 120 -sS -o /tmp/qwen38-vanilla-pp3-nvfp4-one-response.json -w '%{http_code}' \
  "http://127.0.0.1:${PORT}/generate" -H 'Content-Type: application/json' \
  --data-binary @/tmp/qwen38-vanilla-pp3-nvfp4-one.json)"
echo "single_http=$HTTP"
python3 - <<'PY'
import json
try:
    d=json.load(open('/tmp/qwen38-vanilla-pp3-nvfp4-one-response.json'))
    m=d.get('meta_info',{})
    print('single_prompt_tokens=',m.get('prompt_tokens'))
    print('single_completion_tokens=',m.get('completion_tokens'))
except Exception as e:
    print('single_parse_error=',repr(e))
PY

echo '=== GPU MEMORY AFTER STARTUP ==='
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,pstate --format=csv

echo '============================================================'
echo ' INTERPRETATION'
echo " - >= ${REQUIRED}: vanilla PP3 can physically hold raw 3x256K with NVFP4 KV (MTP OFF)."
echo " - <  ${REQUIRED}: even NVFP4 PP3 at this budget cannot physically hold raw 3x256K."
echo ' - This does NOT prove PP3+MTP: stock SGLang rejects PP+EAGLE, and target NVFP4 previously broke EAGLE acceptance.'
echo ' - Do NOT sum PP-stage capacities; every stage must cover all live token positions for its own layers.'
echo '============================================================'
