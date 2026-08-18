#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-sglang:qwen38-27b-pp-mtp-share}"
MODEL="${MODEL:-gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090}"
MTP_CONTAINER="${MTP_CONTAINER:-sglang-qwen38-gittensor-pp3}"
MTP_PORT="${MTP_PORT:-30001}"
TARGET_CONTAINER="${TARGET_CONTAINER:-sglang-qwen38-gittensor-pp3-targetonly}"
TARGET_PORT="${TARGET_PORT:-30002}"
PARTITION="${PARTITION:-23,28,13}"
CTX="${CTX:-262144}"
MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-524288}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.84}"
CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-1024}"
ROOT="${ROOT:-/tmp/qwen38-mm-target-ab}"
RESTORE_MTP="${RESTORE_MTP:-1}"
CG_JSON='{"decode":{"backend":"full","max_bs":2,"bs":[1,2]},"prefill":{"backend":"disabled"}}'
mkdir -p "$ROOT"

restore_mtp() {
  docker rm -f "$TARGET_CONTAINER" >/dev/null 2>&1 || true
  if [[ "$RESTORE_MTP" == "1" ]]; then
    echo "=== RESTORE VALIDATED MTP SERVER ==="
    IMAGE="$IMAGE" MODEL="$MODEL" PORT="$MTP_PORT" \
      bash tools/qwen38_gittensor_pp3_nvfp4_256k_x2_serve.sh || true
  fi
}
trap restore_mtp EXIT

if ! curl -fsS --max-time 5 "http://127.0.0.1:${MTP_PORT}/model_info" >/dev/null; then
  echo "MM_AB_MTP_SERVER_READY=False"
  echo "Start the validated MTP serve preset first." >&2
  exit 2
fi
echo "MM_AB_MTP_SERVER_READY=True"

# Text-only control on the currently running MTP server. This distinguishes a
# broad decode regression from a path that is specific to multimodal prefill.
python3 - "$ROOT/text_control.json" "$MODEL" <<'PY'
import json, sys
out, model = sys.argv[1], sys.argv[2]
req = {
    "model": model,
    "messages": [{"role": "user", "content": "Reply with exactly: TEXT_CONTROL_OK"}],
    "temperature": 0,
    "max_tokens": 64,
}
json.dump(req, open(out, "w"), separators=(",", ":"))
PY
curl -sS --max-time 120 \
  -o "$ROOT/text_control_response.json" \
  "http://127.0.0.1:${MTP_PORT}/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  --data-binary "@$ROOT/text_control.json"
TEXT_OK="$(python3 - "$ROOT/text_control_response.json" <<'PY'
import json, re, sys
d = json.load(open(sys.argv[1]))
choice = (d.get("choices") or [{}])[0]
msg = choice.get("message") or {}
content = str(msg.get("content") or "").strip()
reasoning = str(msg.get("reasoning_content") or "").strip()
combined = " ".join(x for x in (reasoning, content) if x)
print("1" if "TEXT_CONTROL_OK" in combined else "0")
print(f"mtp_text_content={content!r}", file=sys.stderr)
print(f"mtp_text_reasoning={reasoning!r}", file=sys.stderr)
PY
)"
echo "mtp_text_control_pass=$([[ "$TEXT_OK" == "1" ]] && echo True || echo False)"

# Stop production MTP only after the text control has been captured.
docker rm -f "$MTP_CONTAINER" >/dev/null 2>&1 || true
docker rm -f "$TARGET_CONTAINER" >/dev/null 2>&1 || true

# Launch the exact same target model / PP partition / KV / graph settings, but
# with speculative decoding completely absent. No MTP worker is instantiated.
echo "=== TARGET-ONLY PP3 VLM CONTROL ==="
docker run -d \
  --name "$TARGET_CONTAINER" \
  --gpus '"device=0,2,1"' \
  --ipc=host \
  --shm-size 32g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -e "SGLANG_PP_LAYER_PARTITION=${PARTITION}" \
  -e "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" \
  -p "${TARGET_PORT}:30000" \
  -v sglang-hf-cache:/root/.cache/huggingface \
  "$IMAGE" \
  python3 -m sglang.launch_server \
    --model-path "$MODEL" \
    --tp-size 1 \
    --pp-size 3 \
    --trust-remote-code \
    --context-length "$CTX" \
    --max-total-tokens "$MAX_TOTAL_TOKENS" \
    --skip-server-warmup \
    --kv-cache-dtype nvfp4 \
    --attention-backend flashinfer \
    --prefill-attention-backend flashinfer \
    --decode-attention-backend trtllm_mha \
    --page-size 64 \
    --mem-fraction-static "$MEM_FRACTION_STATIC" \
    --max-running-requests 2 \
    --max-mamba-cache-size 2 \
    --mamba-ssm-dtype bfloat16 \
    --mamba-radix-cache-strategy extra_buffer_lazy \
    --disable-radix-cache \
    --mm-feature-transport cpu \
    --chunked-prefill-size "$CHUNKED_PREFILL_SIZE" \
    --cuda-graph-config "$CG_JSON" \
    --disable-flashinfer-autotune \
    --reasoning-parser qwen3 \
    --tool-call-parser qwen3_coder \
    --host 0.0.0.0 \
    --port 30000 >/dev/null

if ! timeout 300 bash -c '
while true; do
  if curl -fsS http://127.0.0.1:'"$TARGET_PORT"'/model_info >/dev/null 2>&1; then exit 0; fi
  running="$(docker inspect -f "{{.State.Running}}" '"$TARGET_CONTAINER"' 2>/dev/null || echo false)"
  if [[ "$running" != true ]]; then exit 2; fi
  sleep 1
done
'; then
  echo "MM_AB_TARGET_BOOT_PASS=False"
  docker logs "$TARGET_CONTAINER" 2>&1 | tail -300 || true
  exit 3
fi
echo "MM_AB_TARGET_BOOT_PASS=True"

set +e
PORT="$TARGET_PORT" CONTAINER="$TARGET_CONTAINER" ROOT="$ROOT/target_mm" \
  bash tools/qwen38_gittensor_pp3_nvfp4_mm_smoke.sh
TARGET_MM_RC=$?
set -e
TARGET_MM_OK=False
if (( TARGET_MM_RC == 0 )); then TARGET_MM_OK=True; fi
echo "target_only_mm_pass=${TARGET_MM_OK}"

if [[ "$TEXT_OK" != "1" ]]; then
  echo "MM_AB_CLASSIFICATION=BROAD_TEXT_DECODE_REGRESSION"
elif [[ "$TARGET_MM_OK" == "True" ]]; then
  echo "MM_AB_CLASSIFICATION=NATIVE_MTP_MM_SEMANTIC_PATH"
else
  echo "MM_AB_CLASSIFICATION=TARGET_PP3_VLM_OR_MODEL_PATH"
fi

echo "QWEN38_GITTENSOR_PP3_NVFP4_MM_TARGET_AB=PASS"
