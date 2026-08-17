#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-lmsysorg/sglang:qwen38-27b}"
MODEL="${MODEL:-RadixArk/Qwen3.8-27B-NVFP4}"
PORT="${PORT:-30000}"
CTX="${CTX:-262144}"
MAX_RUNNING="${MAX_RUNNING:-3}"
PARTITION="${PARTITION:-20,24,20}"
REQUIRED=$((CTX * MAX_RUNNING))
CONTAINER="sglang-qwen38-vanilla-pp3-balanced"

cleanup() {
  docker rm -f \
    sglang-qwen38-test \
    sglang-qwen38-vanilla-pp3 \
    sglang-qwen38-vanilla-pp3-capacity \
    sglang-qwen38-vanilla-pp3-nvfp4-capacity \
    "$CONTAINER" >/dev/null 2>&1 || true
}

important_logs() {
  docker logs "$CONTAINER" 2>&1 | \
    grep -Ei 'SGLANG_PP_LAYER_PARTITION|pipeline|PP[0-9]|NVFP4|FP4|KV Cache|#tokens|max_total_num_tokens|Memory pool|available_gpu_mem|Mamba|running-req|queue-req|Traceback|AssertionError|RuntimeError|ValueError|CUDA error|out of memory|NCCL|exception' | \
    tail -700 || true
}

parse_capacity() {
  local log="$1"
  python3 - "$log" "$REQUIRED" <<'PY'
import re, sys
p, required = sys.argv[1], int(sys.argv[2])
text = open(p, errors='replace').read()
vals=[]
for pat in [r'max_total_num_tokens[^0-9]*([0-9]+)', r'#tokens:\s*([0-9]+)']:
    vals.extend(int(x) for x in re.findall(pat, text, flags=re.I))
uniq=[]
for v in vals:
    if v not in uniq:
        uniq.append(v)
if not uniq:
    print('CAPACITY=0')
    print('parsed_token_capacities=NONE')
    raise SystemExit
mn=min(uniq); mx=max(uniq)
print(f'CAPACITY={mn}')
print('parsed_token_capacities=' + ','.join(map(str,uniq)))
print(f'min_reported_capacity={mn}')
print(f'max_reported_capacity={mx}')
print(f'required_for_3x256k={required}')
print(f'capacity_ratio_to_3x256k={mn/required:.4f}')
print(f'raw_3x256k_capacity_gate={mn >= required}')
print(f'equal_share_tokens_per_session={mn//3}')
PY
}

start_case() {
  local mem="$1"
  cleanup
  sleep 2
  echo
  echo "=== START BALANCED VANILLA PP3: partition=${PARTITION} mem_fraction=${mem} ==="
  docker run -d \
    --name "$CONTAINER" \
    --gpus '"device=0,2,1"' \
    --ipc=host \
    --shm-size 32g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -e "SGLANG_PP_LAYER_PARTITION=${PARTITION}" \
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
      --mem-fraction-static "$mem" \
      --max-running-requests "$MAX_RUNNING" \
      --max-mamba-cache-size 8 \
      --mamba-ssm-dtype bfloat16 \
      --mamba-radix-cache-strategy extra_buffer_lazy \
      --disable-radix-cache \
      --mm-feature-transport cpu \
      --chunked-prefill-size 2048 \
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
    echo "BOOT_FAIL mem_fraction=${mem}"
    important_logs
    return 1
  fi

  local log="/tmp/qwen38-balanced-${mem}.log"
  docker logs "$CONTAINER" >"$log" 2>&1 || true
  echo "BOOT_OK mem_fraction=${mem}"
  grep -Ei 'PP[0-9].*(Mamba Cache|KV Cache|Memory pool end|max_total_num_tokens)' "$log" | tail -80 || true
  parse_capacity "$log"
}

echo '============================================================'
echo ' QWEN3.8 VANILLA PP3 + NVFP4 BALANCED-PARTITION SMOKE'
echo " partition=${PARTITION}"
echo " required=${REQUIRED} token positions for 3 x ${CTX}"
echo ' MTP OFF / NO SOURCE MOUNT / CUDA GRAPH OFF'
echo '============================================================'

echo '=== EXACT-IMAGE PARTITION ENV PREFLIGHT ==='
# The exact image must already support the manual PP partition env.  This is a
# vanilla runtime feature; no source patch is mounted.
docker run --rm \
  -e "SGLANG_PP_LAYER_PARTITION=${PARTITION}" \
  "$IMAGE" \
  python3 - <<'PY'
import os
from sglang.srt.distributed import get_pp_indices
p=os.environ['SGLANG_PP_LAYER_PARTITION']
parts=[int(x) for x in p.split(',')]
assert sum(parts)==64, (parts, sum(parts))
assert len(parts)==3, parts
spans=[get_pp_indices(64,r,3) for r in range(3)]
print('partition_env=',p)
print('pp_spans=',spans)
assert spans == [(0,parts[0]),(parts[0],parts[0]+parts[1]),(parts[0]+parts[1],64)], spans
print('partition_preflight=PASS')
PY

echo '=== GPU STATE BEFORE ==='
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,pstate --format=csv

CAP=0
if start_case 0.95; then
  CAP="$(python3 - <<'PY'
import re
p='/tmp/qwen38-balanced-0.95.log'
t=open(p,errors='replace').read()
v=[int(x) for x in re.findall(r'max_total_num_tokens[^0-9]*([0-9]+)',t,re.I)]
print(min(v) if v else 0)
PY
)"
fi

if (( CAP < REQUIRED )); then
  echo
  echo "0.95 capacity ${CAP} < ${REQUIRED}; retrying only once at 0.99 ceiling..."
  CAP=0
  if start_case 0.99; then
    CAP="$(python3 - <<'PY'
import re
p='/tmp/qwen38-balanced-0.99.log'
t=open(p,errors='replace').read()
v=[int(x) for x in re.findall(r'max_total_num_tokens[^0-9]*([0-9]+)',t,re.I)]
print(min(v) if v else 0)
PY
)"
  fi
fi

echo
echo '=== FINAL CAPACITY VERDICT ==='
echo "partition=${PARTITION}"
echo "final_capacity=${CAP}"
echo "required=${REQUIRED}"
if (( CAP >= REQUIRED )); then
  echo 'BALANCED_RAW_3X256K_CAPACITY=PASS'
else
  echo 'BALANCED_RAW_3X256K_CAPACITY=FAIL'
fi

echo '=== ROBUST SINGLE REQUEST CHECK ==='
python3 - <<'PY'
import json
json.dump({
  'input_ids':[1000]*256,
  'sampling_params':{'temperature':0,'max_new_tokens':8,'ignore_eos':True},
}, open('/tmp/qwen38-balanced-one.json','w'))
PY
set +e
HTTP="$(curl --max-time 60 -sS -o /tmp/qwen38-balanced-one-response.json -w '%{http_code}' \
  "http://127.0.0.1:${PORT}/generate" -H 'Content-Type: application/json' \
  --data-binary @/tmp/qwen38-balanced-one.json)"
CURL_RC=$?
set -e
echo "single_curl_rc=${CURL_RC}"
echo "single_http=${HTTP}"
if [[ "$CURL_RC" -eq 0 && "$HTTP" == 200 ]]; then
  python3 - <<'PY'
import json
d=json.load(open('/tmp/qwen38-balanced-one-response.json'))
m=d.get('meta_info',{})
print('single_prompt_tokens=',m.get('prompt_tokens'))
print('single_completion_tokens=',m.get('completion_tokens'))
print('single_functional_pass=', int(m.get('prompt_tokens') or 0)==256 and int(m.get('completion_tokens') or 0)==8)
PY
else
  echo 'single_functional_pass=False'
  echo '=== REQUEST FAILURE / HANG LOGS ==='
  important_logs
fi

echo '=== GPU STATE AFTER ==='
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,pstate --format=csv || true

echo '============================================================'
echo 'Interpretation:'
echo '  capacity PASS + request PASS: vanilla PP3+NVFP4 physically solves raw 3x256K (MTP still OFF).'
echo '  capacity PASS + request FAIL: memory fits, but vanilla PP3 NVFP4 execution path is unusable in this build.'
echo '  capacity FAIL: balanced vanilla PP3 still cannot physically hold raw 3x256K.'
echo '  This test does NOT claim MTP support; stock PP+EAGLE is separately rejected.'
echo '============================================================'
