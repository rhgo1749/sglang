#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-lmsysorg/sglang:qwen38-27b}"
MODEL="${MODEL:-RadixArk/Qwen3.8-27B-NVFP4}"
CONTAINER="${CONTAINER:-sglang-qwen38-vanilla-pp3-final}"
PORT="${PORT:-30000}"
CTX="${CTX:-262144}"
MAX_RUNNING="${MAX_RUNNING:-3}"
MAX_MAMBA="${MAX_MAMBA:-3}"
# 19,23,22 is the only split so far that clears the raw 3x256K capacity gate.
PARTITION="${PARTITION:-19,23,22}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.99}"
CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-2048}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
REQUIRED=$((CTX * MAX_RUNNING))

cleanup() {
  docker rm -f \
    sglang-qwen38-test \
    sglang-qwen38-vanilla-pp3 \
    sglang-qwen38-vanilla-pp3-capacity \
    sglang-qwen38-vanilla-pp3-nvfp4-capacity \
    sglang-qwen38-vanilla-pp3-balanced \
    "$CONTAINER" >/dev/null 2>&1 || true
}

dump_logs() {
  docker logs "$CONTAINER" 2>&1 | \
    grep -Ei 'PP[0-9]|pipeline|NVFP4|KV Cache|Mamba Cache|Memory pool end|max_total_num_tokens|available_gpu_mem|running-req|queue-req|Traceback|AssertionError|RuntimeError|CUDA error|out of memory|exception' | \
    tail -600 || true
}

echo '============================================================'
echo ' QWEN3.8 FINAL VANILLA PP3 + NVFP4 RAW 3x256K GATE'
echo " partition=${PARTITION}"
echo " mem_fraction_static=${MEM_FRACTION_STATIC}"
echo " chunked_prefill_size=${CHUNKED_PREFILL_SIZE}"
echo " pytorch_cuda_alloc_conf=${PYTORCH_CUDA_ALLOC_CONF}"
echo " max_running_requests=${MAX_RUNNING}"
echo " max_mamba_cache_size=${MAX_MAMBA}"
echo " required=${REQUIRED}"
echo ' MTP OFF / CUDA GRAPH OFF / NO SOURCE MOUNTS'
echo '============================================================'

if (( MAX_MAMBA < MAX_RUNNING )); then
  echo "ERROR: max_mamba_cache_size (${MAX_MAMBA}) must cover max_running_requests (${MAX_RUNNING}) in this no-radix/no-overlap gate"
  exit 64
fi

cleanup
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
  dump_logs
  exit 1
fi

echo 'BOOT_PASS=True'
LOG="/tmp/qwen38-final-pp3.log"
docker logs "$CONTAINER" >"$LOG" 2>&1 || true

grep -Ei 'PP[0-9].*(Mamba Cache|KV Cache|Memory pool end|max_total_num_tokens)' "$LOG" | tail -100 || true

CAP="$(python3 - "$LOG" <<'PY'
import re,sys
t=open(sys.argv[1],errors='replace').read()
v=[int(x) for x in re.findall(r'max_total_num_tokens[^0-9]*([0-9]+)',t,re.I)]
print(min(v) if v else 0)
PY
)"

SHORT=$(( REQUIRED - CAP ))
if (( SHORT < 0 )); then SHORT=0; fi

echo "final_capacity=${CAP}"
echo "required=${REQUIRED}"
echo "capacity_shortfall=${SHORT}"
python3 - "$CAP" "$REQUIRED" <<'PY'
import sys
cap,req=map(int,sys.argv[1:])
print(f'capacity_ratio={cap/req:.6f}')
print(f'equal_share_tokens_per_session={cap//3}')
print(f'raw_3x256k_capacity_gate={cap>=req}')
PY

python3 - "$LOG" <<'PY'
import re,sys
text=open(sys.argv[1],errors='replace').read()
vals={}
for pp,mem in re.findall(r'PP(\d+).*?available_gpu_mem=([0-9.]+) GB',text):
    vals[int(pp)]=float(mem)
for pp in sorted(vals):
    print(f'pp{pp}_runtime_headroom_gb={vals[pp]:.2f}')
if vals:
    print(f'min_runtime_headroom_gb={min(vals.values()):.2f}')
else:
    print('min_runtime_headroom_gb=unknown')
PY

if (( CAP < REQUIRED )); then
  echo 'FINAL_RAW_3X256K_CAPACITY=FAIL'
  echo '=== GPU STATE ==='
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,pstate --format=csv || true
  exit 2
fi

echo 'FINAL_RAW_3X256K_CAPACITY=PASS'

echo '=== SINGLE FUNCTIONAL REQUEST ==='
python3 - <<'PY'
import json
json.dump({
  'input_ids':[1000]*256,
  'sampling_params':{'temperature':0,'max_new_tokens':8,'ignore_eos':True},
},open('/tmp/qwen38-final-one.json','w'))
PY
set +e
HTTP="$(curl --max-time 60 -sS -o /tmp/qwen38-final-one-response.json -w '%{http_code}' \
  "http://127.0.0.1:${PORT}/generate" -H 'Content-Type: application/json' \
  --data-binary @/tmp/qwen38-final-one.json)"
RC=$?
set -e
echo "single_curl_rc=${RC}"
echo "single_http=${HTTP}"
if [[ "$RC" -eq 0 && "$HTTP" == 200 ]]; then
  python3 - <<'PY'
import json
d=json.load(open('/tmp/qwen38-final-one-response.json'))
m=d.get('meta_info',{})
ok=int(m.get('prompt_tokens') or 0)==256 and int(m.get('completion_tokens') or 0)==8
print('single_prompt_tokens=',m.get('prompt_tokens'))
print('single_completion_tokens=',m.get('completion_tokens'))
print('single_functional_pass=',ok)
raise SystemExit(0 if ok else 1)
PY
else
  echo 'single_functional_pass=False'
  dump_logs
  exit 3
fi

echo '=== PARALLEL-3 MAMBA SLOT CHECK ==='
for i in 0 1 2; do
  python3 - "$i" <<'PY'
import json,sys
i=int(sys.argv[1])
json.dump({
  'input_ids':[1100+i]*256,
  'sampling_params':{'temperature':0,'max_new_tokens':8,'ignore_eos':True},
},open(f'/tmp/qwen38-final-p3-{i}.json','w'))
PY
done
PIDS=()
for i in 0 1 2; do
  (
    curl --max-time 90 -sS \
      -o "/tmp/qwen38-final-p3-${i}-response.json" \
      -w '%{http_code}' \
      "http://127.0.0.1:${PORT}/generate" \
      -H 'Content-Type: application/json' \
      --data-binary "@/tmp/qwen38-final-p3-${i}.json" \
      >"/tmp/qwen38-final-p3-${i}-http.txt"
  ) &
  PIDS+=("$!")
done
P3_RC=0
for pid in "${PIDS[@]}"; do wait "$pid" || P3_RC=1; done
python3 - <<'PY'
import json,pathlib
ok=True
for i in range(3):
    hp=pathlib.Path(f'/tmp/qwen38-final-p3-{i}-http.txt')
    http=hp.read_text().strip() if hp.exists() else ''
    print(f'req{i}_http={http}')
    try:
        d=json.load(open(f'/tmp/qwen38-final-p3-{i}-response.json'))
        m=d.get('meta_info',{})
        p=int(m.get('prompt_tokens') or 0)
        c=int(m.get('completion_tokens') or 0)
        print(f'req{i}_prompt_tokens={p}')
        print(f'req{i}_completion_tokens={c}')
        ok &= http=='200' and p==256 and c==8
    except Exception as e:
        print(f'req{i}_parse_error={e!r}')
        ok=False
print(f'parallel3_mamba_slot_pass={ok}')
raise SystemExit(0 if ok else 1)
PY

echo '=== GPU STATE ==='
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,pstate --format=csv || true

echo 'VANILLA_PP3_NVFP4_RAW_3X256K_FINAL_GATE=PASS'
