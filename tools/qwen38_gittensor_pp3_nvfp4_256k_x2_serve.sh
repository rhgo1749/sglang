#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-sglang:qwen38-27b-pp-mtp-share}"
MODEL="${MODEL:-gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090}"
CONTAINER="${CONTAINER:-sglang-qwen38-gittensor-pp3}"
PORT="${PORT:-30001}"

CTX="${CTX:-262144}"
PARTITION="${PARTITION:-23,28,13}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.84}"
MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-524288}"
MAX_RUNNING="${MAX_RUNNING:-2}"
MAX_MAMBA="${MAX_MAMBA:-2}"
CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-1024}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
SEMANTIC_BOOT_CHECK="${SEMANTIC_BOOT_CHECK:-1}"

if [[ "$PARTITION" != "23,28,13" ]]; then
  echo "WARNING: validated partition is 23,28,13; requested ${PARTITION}" >&2
fi
if (( MAX_TOTAL_TOKENS < 524288 )); then
  echo "ERROR: validated 2x256K preset requires max_total_tokens >= 524288" >&2
  exit 64
fi
if (( MAX_RUNNING != 2 || MAX_MAMBA != 2 )); then
  echo "ERROR: validated preset requires max_running_requests=max_mamba_cache_size=2" >&2
  exit 65
fi

docker rm -f "$CONTAINER" >/dev/null 2>&1 || true

echo "Starting correctness-validated Qwen3.8 PP3/native-MTP 2x256K preset"
echo "  image=${IMAGE}"
echo "  model=${MODEL}"
echo "  port=${PORT}"
echo "  partition=${PARTITION}"
echo "  context=${CTX}"
echo "  pool=${MAX_TOTAL_TOKENS}"
echo "  chunked_prefill=${CHUNKED_PREFILL_SIZE}"
echo "  target_verify_graph=disabled (capture/replay eager)"
echo "  prefill_graph=disabled (post-long semantic stop condition)"
echo "  eagle_draft_decode_graph=enabled"
echo "  eagle_draft_extend_graph=enabled"
echo "  semantic_boot_check=${SEMANTIC_BOOT_CHECK}"

docker run -d \
  --name "$CONTAINER" \
  --gpus '"device=0,2,1"' \
  --ipc=host \
  --shm-size 32g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -e "SGLANG_PP_LAYER_PARTITION=${PARTITION}" \
  -e "PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF}" \
  -e "SGLANG_MTP_DISABLE_TARGET_VERIFY_CUDA_GRAPH=1" \
  -e "SGLANG_MTP_DISABLE_DRAFT_DECODE_CUDA_GRAPH=0" \
  -e "SGLANG_DISABLE_DRAFT_EXTEND_CUDA_GRAPH=0" \
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
    --max-running-requests "$MAX_RUNNING" \
    --max-mamba-cache-size "$MAX_MAMBA" \
    --mamba-ssm-dtype bfloat16 \
    --mamba-radix-cache-strategy extra_buffer_lazy \
    --disable-radix-cache \
    --mm-feature-transport cpu \
    --chunked-prefill-size "$CHUNKED_PREFILL_SIZE" \
    --cuda-graph-config '{"decode":{"backend":"full","max_bs":2,"bs":[1,2]},"prefill":{"backend":"disabled"}}' \
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
  echo "BOOT_PASS=False"
  docker logs "$CONTAINER" 2>&1 | tail -300 || true
  exit 1
fi

echo "BOOT_PASS=True"

if [[ "$SEMANTIC_BOOT_CHECK" == "1" ]]; then
  REQ="$(mktemp)"
  RESP="$(mktemp)"
  trap 'rm -f "$REQ" "$RESP"' EXIT
  python3 - "$REQ" "$MODEL" <<'PY'
import json, sys
json.dump({
    "model": sys.argv[2],
    "messages": [{"role": "user", "content": "Reply with exactly: TEXT_CONTROL_OK"}],
    "temperature": 0,
    "max_tokens": 256,
}, open(sys.argv[1], "w"), separators=(",", ":"))
PY
  set +e
  HTTP="$(curl --max-time 180 -sS -o "$RESP" -w '%{http_code}' \
    "http://127.0.0.1:${PORT}/v1/chat/completions" \
    -H 'Content-Type: application/json' --data-binary "@$REQ")"
  RC=$?
  set -e
  if ! python3 - "$RESP" "$HTTP" "$RC" <<'PY'
import json, sys
path, http, rc = sys.argv[1], sys.argv[2], int(sys.argv[3])
ok = rc == 0 and http == "200"
try:
    d = json.load(open(path))
    c = (d.get("choices") or [{}])[0]
    m = c.get("message") or {}
    combined = " ".join(
        x for x in (
            str(m.get("reasoning_content") or ""),
            str(m.get("content") or ""),
        ) if x
    )
    ok = ok and "TEXT_CONTROL_OK" in combined
except Exception:
    ok = False
print(f"SEMANTIC_BOOT_PASS={ok}")
raise SystemExit(0 if ok else 1)
PY
  then
    echo "ERROR: server booted but native-MTP semantic control failed" >&2
    docker logs "$CONTAINER" 2>&1 | tail -240 || true
    docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
    exit 3
  fi
fi

echo "QWEN38_GITTENSOR_PP3_NVFP4_X2_256K_SERVE=READY"
echo "endpoint=http://127.0.0.1:${PORT}"
