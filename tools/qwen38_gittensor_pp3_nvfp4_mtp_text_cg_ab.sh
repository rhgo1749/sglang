#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-sglang:qwen38-27b-pp-mtp-share}"
MODEL="${MODEL:-gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090}"
CONTAINER="${CONTAINER:-sglang-qwen38-gittensor-pp3}"
PORT="${PORT:-30001}"
PARTITION="${PARTITION:-23,28,13}"
CTX="${CTX:-262144}"
MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-524288}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.84}"
CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-1024}"
ROOT="${ROOT:-/tmp/qwen38-mtp-text-cg-ab}"
RESTORE_MTP="${RESTORE_MTP:-1}"
CG_JSON='{"decode":{"backend":"full","max_bs":2,"bs":[1,2]},"prefill":{"backend":"disabled"}}'
mkdir -p "$ROOT"

restore_mtp() {
  docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
  if [[ "$RESTORE_MTP" == "1" ]]; then
    echo "=== RESTORE VALIDATED MTP SERVER ==="
    IMAGE="$IMAGE" MODEL="$MODEL" PORT="$PORT" \
      bash tools/qwen38_gittensor_pp3_nvfp4_256k_x2_serve.sh || true
  fi
}
trap restore_mtp EXIT

launch_mtp() {
  local mode="$1"
  docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
  local graph_args=()
  if [[ "$mode" == "on" ]]; then
    graph_args=(--cuda-graph-config "$CG_JSON")
  else
    graph_args=(--disable-cuda-graph)
  fi

  docker run -d \
    --name "$CONTAINER" \
    --gpus '"device=0,2,1"' \
    --ipc=host \
    --shm-size 32g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -e "SGLANG_PP_LAYER_PARTITION=${PARTITION}" \
    -e "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" \
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
      --max-running-requests 2 \
      --max-mamba-cache-size 2 \
      --mamba-ssm-dtype bfloat16 \
      --mamba-radix-cache-strategy extra_buffer_lazy \
      --disable-radix-cache \
      --mm-feature-transport cpu \
      --chunked-prefill-size "$CHUNKED_PREFILL_SIZE" \
      "${graph_args[@]}" \
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
    echo "mtp_text_cg_${mode}_boot_pass=False"
    docker logs "$CONTAINER" 2>&1 | tail -300 || true
    return 1
  fi
  echo "mtp_text_cg_${mode}_boot_pass=True"
}

text_control() {
  local mode="$1"
  local req="$ROOT/${mode}-request.json"
  local resp="$ROOT/${mode}-response.json"
  python3 - "$req" "$MODEL" <<'PY'
import json, sys
out, model = sys.argv[1], sys.argv[2]
json.dump({
    "model": model,
    "messages": [{"role": "user", "content": "Reply with exactly: TEXT_CONTROL_OK"}],
    "temperature": 0,
    "max_tokens": 256,
}, open(out, "w"), separators=(",", ":"))
PY

  local http rc
  set +e
  http="$(curl --max-time 180 -sS -o "$resp" -w '%{http_code}' \
    "http://127.0.0.1:${PORT}/v1/chat/completions" \
    -H 'Content-Type: application/json' --data-binary "@$req")"
  rc=$?
  set -e

  python3 - "$resp" "$http" "$rc" "$mode" <<'PY'
import json, pathlib, sys
path, http, rc, mode = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
ok = rc == 0 and http == "200"
content = reasoning = finish = ""
try:
    d = json.load(open(path))
    choice = (d.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    content = str(msg.get("content") or "").strip()
    reasoning = str(msg.get("reasoning_content") or "").strip()
    finish = str(choice.get("finish_reason") or "")
    combined = " ".join(x for x in (reasoning, content) if x)
    semantic = "TEXT_CONTROL_OK" in combined
    ok = ok and semantic
    print(f"mtp_text_cg_{mode}_http={http}")
    print(f"mtp_text_cg_{mode}_content={content!r}")
    print(f"mtp_text_cg_{mode}_reasoning={reasoning!r}")
    print(f"mtp_text_cg_{mode}_finish_reason={finish!r}")
    print(f"mtp_text_cg_{mode}_semantic_pass={semantic}")
except Exception as e:
    print(f"mtp_text_cg_{mode}_parse_error={e!r}")
    ok = False
print(f"mtp_text_cg_{mode}_pass={ok}")
pathlib.Path(path + ".pass").write_text("1" if ok else "0")
PY
}

echo "=== MTP TEXT SEMANTIC CONTROL: CUDA GRAPH OFF ==="
launch_mtp off
text_control off
OFF="$(cat "$ROOT/off-response.json.pass")"

echo "=== MTP TEXT SEMANTIC CONTROL: DECODE CUDA GRAPH BS=1,2 ==="
launch_mtp on
text_control on
ON="$(cat "$ROOT/on-response.json.pass")"

if [[ "$OFF" == "1" && "$ON" == "0" ]]; then
  echo "MTP_TEXT_CG_AB_CLASSIFICATION=MTP_DECODE_CUDA_GRAPH_SEMANTIC_REGRESSION"
elif [[ "$OFF" == "0" && "$ON" == "0" ]]; then
  echo "MTP_TEXT_CG_AB_CLASSIFICATION=MTP_CORE_SEMANTIC_REGRESSION"
elif [[ "$OFF" == "1" && "$ON" == "1" ]]; then
  echo "MTP_TEXT_CG_AB_CLASSIFICATION=NO_REPRO_OR_TRANSIENT_SERVER_STATE"
else
  echo "MTP_TEXT_CG_AB_CLASSIFICATION=UNEXPECTED_CG_OFF_FAIL_ON_PASS"
fi

echo "QWEN38_GITTENSOR_PP3_NVFP4_MTP_TEXT_CG_AB=PASS"
