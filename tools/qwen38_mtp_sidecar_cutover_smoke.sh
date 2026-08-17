#!/usr/bin/env bash
set -euo pipefail

REPO="${HOME}/projects/sglang-fork"
PATCH_DIR="${HOME}/projects/sglang-patches"
CONTAINER="sglang-qwen38-test"
IMAGE="lmsysorg/sglang:qwen38-27b"
MODEL="RadixArk/Qwen3.8-27B-NVFP4"
# With the colocated draft removed, 0.95 lets the target KV pool consume nearly
# all reclaimed VRAM before the target-verify CUDA graph is captured. Keep a
# conservative graph/eager workspace reserve for the authoritative cutover gate.
# The runtime cutover itself still fail-fast checks target/sidecar pools >=64K.
MEM_FRACTION_STATIC="${MTP_CUTOVER_MEM_FRACTION_STATIC:-0.80}"

cd "$REPO"

echo '=== APPLY CUTOVER ==='
python3 tools/qwen38_mtp_sidecar_cutover.py --commit

echo '=== HOTFIX NESTED PP CONTEXT ==='
python3 tools/qwen38_mtp_cutover_pp_hotfix.py

echo "=== RECREATE SERVER (mem_fraction_static=${MEM_FRACTION_STATIC}) ==="
docker rm -f "$CONTAINER" >/dev/null 2>&1 || true

docker run -d \
  --name "$CONTAINER" \
  --gpus '"device=0,2,1"' \
  --ipc=host \
  --shm-size 32g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -p 30000:30000 \
  -v sglang-hf-cache:/root/.cache/huggingface \
  -v "$PATCH_DIR/eagle_worker_v2.sidecar-pool-probe.py:/sgl-workspace/sglang/python/sglang/srt/speculative/eagle_worker_v2.py:ro" \
  -v "$PATCH_DIR/qwen3_5_mtp.sidecar.py:/sgl-workspace/sglang/python/sglang/srt/models/qwen3_5_mtp.py:ro" \
  "$IMAGE" \
  python3 -m sglang.launch_server \
    --model-path "$MODEL" \
    --tp 2 \
    --trust-remote-code \
    --context-length 262144 \
    --mem-fraction-static "$MEM_FRACTION_STATIC" \
    --max-running-requests 1 \
    --max-mamba-cache-size 8 \
    --mamba-ssm-dtype bfloat16 \
    --mamba-radix-cache-strategy extra_buffer_lazy \
    --disable-radix-cache \
    --mm-feature-transport cpu \
    --attention-backend flashinfer \
    --chunked-prefill-size 2048 \
    --speculative-algorithm EAGLE \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 4 \
    --reasoning-parser qwen3 \
    --tool-call-parser qwen3_coder \
    --host 0.0.0.0 \
    --port 30000 >/dev/null

STARTED="$(docker inspect -f '{{.State.StartedAt}}' "$CONTAINER")"
echo "StartedAt: $STARTED"

dump_runtime_failure() {
  echo '=== REQUEST FAILURE CUTOVER SUMMARY ==='
  docker logs --since "$STARTED" "$CONTAINER" 2>&1 | \
    grep -E 'MTP-CUTOVER|MTP-SIDECAR|Prefill batch|Decode batch|Scheduler hit an exception|Traceback|Exception|RuntimeError|AssertionError|CUDA error|illegal memory|out of memory' | \
    tail -320 || true
  echo '=== REQUEST FAILURE TAIL ==='
  docker logs --since "$STARTED" "$CONTAINER" 2>&1 | tail -260 || true
  echo '=== CONTAINER STATUS ==='
  docker inspect -f 'running={{.State.Running}} status={{.State.Status}} exit={{.State.ExitCode}}' "$CONTAINER" || true
}

echo '=== WAIT FOR SERVER ==='
if ! timeout 180 bash -c '
while true; do
  if curl -fsS http://127.0.0.1:30000/model_info >/dev/null 2>&1; then
    exit 0
  fi
  if [ "$(docker inspect -f "{{.State.Running}}" sglang-qwen38-test 2>/dev/null)" != "true" ]; then
    exit 2
  fi
  sleep 1
done
'; then
  echo 'SERVER DID NOT BECOME READY'
  echo '=== FAILURE MEMORY / CUTOVER SUMMARY ==='
  docker logs --since "$STARTED" "$CONTAINER" 2>&1 | \
    grep -E 'MTP-CUTOVER|Load weight end|Mamba Cache|KV Cache|Memory pool end|max_total_num_tokens|available_gpu_mem|Capture target verify CUDA graph|OutOfMemory|out of memory|Scheduler hit an exception|Traceback' | \
    tail -220 || true
  echo '=== FAILURE TAIL ==='
  docker logs --since "$STARTED" "$CONTAINER" 2>&1 | tail -180
  exit 1
fi

echo '=== STARTUP CUTOVER GATES ==='
docker logs --since "$STARTED" "$CONTAINER" 2>&1 | \
  grep -E 'MTP-CUTOVER|MTP-SIDECAR|target_tokens|max_total_num_tokens|Memory pool end|available_gpu_mem|Scheduler hit an exception|Traceback' | \
  tail -220 || true

echo '=== MULTI-ITERATION DECODE ==='
SHORT_HTTP="$(curl -sS \
  -o /tmp/qwen38-cutover-short.json \
  -w '%{http_code}' \
  http://127.0.0.1:30000/generate \
  -H 'Content-Type: application/json' \
  -d '{
    "text": "Explain in one short paragraph why speculative decoding can improve inference throughput.",
    "sampling_params": {
      "temperature": 0,
      "max_new_tokens": 64
    }
  }' || true)"
echo "short_http=${SHORT_HTTP}"
if [[ "$SHORT_HTTP" != "200" ]]; then
  echo '=== SHORT RESPONSE BODY ==='
  cat /tmp/qwen38-cutover-short.json 2>/dev/null || true
  echo
  dump_runtime_failure
  exit 1
fi
python3 - <<'PY'
import json
p='/tmp/qwen38-cutover-short.json'
d=json.load(open(p))
text=d.get('text','')
meta=d.get('meta_info',{})
print('short_response_chars=', len(text))
print('short_finish_reason=', meta.get('finish_reason'))
print('short_completion_tokens=', meta.get('completion_tokens'))
PY

echo '=== 64K-CLASS TOKEN-ID REQUEST (65520 + 8) ==='
python3 - <<'PY'
import json
payload = {
    "input_ids": [1000] * 65520,
    "sampling_params": {
        "temperature": 0,
        "max_new_tokens": 8,
    },
}
with open('/tmp/qwen38-cutover-64k.json','w') as f:
    json.dump(payload,f,separators=(',',':'))
PY

set +e
/usr/bin/time -f '64k_wall=%e sec' \
  curl -sS \
    -o /tmp/qwen38-cutover-64k-response.json \
    -w '64k_http=%{http_code}\n' \
    http://127.0.0.1:30000/generate \
    -H 'Content-Type: application/json' \
    --data-binary @/tmp/qwen38-cutover-64k.json
CURL64_RC=$?
set -e
if [[ $CURL64_RC -ne 0 ]] || ! python3 - <<'PY'
import json, sys
p='/tmp/qwen38-cutover-64k-response.json'
try:
    d=json.load(open(p))
except Exception:
    sys.exit(1)
if isinstance(d, dict) and (d.get('error') is not None or d.get('detail') is not None):
    sys.exit(1)
PY
then
  echo '=== 64K RESPONSE BODY ==='
  cat /tmp/qwen38-cutover-64k-response.json 2>/dev/null || true
  echo
  dump_runtime_failure
  exit 1
fi

python3 - <<'PY'
import json
p='/tmp/qwen38-cutover-64k-response.json'
d=json.load(open(p))
meta=d.get('meta_info',{})
print('64k_finish_reason=', meta.get('finish_reason'))
print('64k_prompt_tokens=', meta.get('prompt_tokens'))
print('64k_completion_tokens=', meta.get('completion_tokens'))
print('64k_text_chars=', len(d.get('text','')))
PY

echo '=== FINAL CUTOVER LOG ==='
docker logs --since "$STARTED" "$CONTAINER" 2>&1 | \
  grep -E 'MTP-CUTOVER-(POOL|ATTN|GRAPH|ROPE|REQ|PREFILL|DRAFT|EXTEND)|MTP-CUTOVER]|Scheduler hit an exception|CUDA error|illegal memory|Traceback|Prefill batch|Decode batch' | \
  tail -320 || true

echo '=== CONTAINER STATUS ==='
docker inspect -f 'running={{.State.Running}} status={{.State.Status}} exit={{.State.ExitCode}}' "$CONTAINER"

echo 'MTP CUTOVER SMOKE COMPLETE'
