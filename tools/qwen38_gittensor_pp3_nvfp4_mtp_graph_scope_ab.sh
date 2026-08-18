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
ROOT="${ROOT:-/tmp/qwen38-mtp-graph-scope-ab}"
RESTORE_MTP="${RESTORE_MTP:-1}"
CG_JSON='{"decode":{"backend":"full","max_bs":2,"bs":[1,2]},"prefill":{"backend":"disabled"}}'
mkdir -p "$ROOT"

restore_mtp() {
  docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
  if [[ "$RESTORE_MTP" == "1" ]]; then
    echo "=== RESTORE CORRECTNESS BASELINE MTP SERVER ==="
    IMAGE="$IMAGE" MODEL="$MODEL" PORT="$PORT" \
      bash tools/qwen38_gittensor_pp3_nvfp4_256k_x2_serve.sh || true
  fi
}
trap restore_mtp EXIT

launch_mode() {
  local mode="$1"
  docker rm -f "$CONTAINER" >/dev/null 2>&1 || true

  local graph_args=()
  local draft_decode_disable="0"
  local draft_extend_disable="0"
  case "$mode" in
    all_off)
      graph_args=(--disable-cuda-graph)
      ;;
    target_graph_only)
      graph_args=(--cuda-graph-config "$CG_JSON")
      draft_decode_disable="1"
      draft_extend_disable="1"
      ;;
    draft_decode_only)
      graph_args=(--cuda-graph-config "$CG_JSON")
      draft_decode_disable="0"
      draft_extend_disable="1"
      ;;
    draft_extend_only)
      graph_args=(--cuda-graph-config "$CG_JSON")
      draft_decode_disable="1"
      draft_extend_disable="0"
      ;;
    *)
      echo "unknown mode: $mode" >&2
      return 64
      ;;
  esac

  echo "=== ${mode} ==="
  docker run -d \
    --name "$CONTAINER" \
    --gpus '"device=0,2,1"' \
    --ipc=host \
    --shm-size 32g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -e "SGLANG_PP_LAYER_PARTITION=${PARTITION}" \
    -e "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" \
    -e "SGLANG_MTP_DISABLE_DRAFT_DECODE_CUDA_GRAPH=${draft_decode_disable}" \
    -e "SGLANG_DISABLE_DRAFT_EXTEND_CUDA_GRAPH=${draft_extend_disable}" \
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
    echo "mtp_graph_scope_${mode}_boot_pass=False"
    docker logs "$CONTAINER" 2>&1 | tail -300 || true
    return 1
  fi
  echo "mtp_graph_scope_${mode}_boot_pass=True"
  docker logs "$CONTAINER" 2>&1 | grep -E 'MTP-PP-GRAPH-SCOPE|Capture draft (decode|extend) CUDA graph' | tail -40 || true
}

semantic_control() {
  local mode="$1"
  local req="$ROOT/${mode}-request.json"
  local resp="$ROOT/${mode}-response.json"
  python3 - "$req" "$MODEL" <<'PY'
import json, sys
json.dump({
    "model": sys.argv[2],
    "messages": [{"role": "user", "content": "Reply with exactly: TEXT_CONTROL_OK"}],
    "temperature": 0,
    "max_tokens": 256,
}, open(sys.argv[1], "w"), separators=(",", ":"))
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
    print(f"mtp_graph_scope_{mode}_http={http}")
    print(f"mtp_graph_scope_{mode}_content={content!r}")
    print(f"mtp_graph_scope_{mode}_reasoning={reasoning!r}")
    print(f"mtp_graph_scope_{mode}_finish_reason={finish!r}")
    print(f"mtp_graph_scope_{mode}_semantic_pass={semantic}")
except Exception as e:
    print(f"mtp_graph_scope_{mode}_parse_error={e!r}")
    ok = False
print(f"mtp_graph_scope_{mode}_pass={ok}")
pathlib.Path(path + ".pass").write_text("1" if ok else "0")
PY
}

for mode in all_off target_graph_only draft_decode_only draft_extend_only; do
  launch_mode "$mode"
  semantic_control "$mode"
done

OFF="$(cat "$ROOT/all_off-response.json.pass")"
TARGET="$(cat "$ROOT/target_graph_only-response.json.pass")"
DDECODE="$(cat "$ROOT/draft_decode_only-response.json.pass")"
DEXTEND="$(cat "$ROOT/draft_extend_only-response.json.pass")"

echo "mtp_graph_scope_all_off_pass=$([[ "$OFF" == 1 ]] && echo True || echo False)"
echo "mtp_graph_scope_target_graph_only_pass=$([[ "$TARGET" == 1 ]] && echo True || echo False)"
echo "mtp_graph_scope_draft_decode_only_pass=$([[ "$DDECODE" == 1 ]] && echo True || echo False)"
echo "mtp_graph_scope_draft_extend_only_pass=$([[ "$DEXTEND" == 1 ]] && echo True || echo False)"

if [[ "$OFF" != 1 ]]; then
  CLASS="BASELINE_SEMANTIC_FAILURE"
elif [[ "$TARGET" != 1 ]]; then
  CLASS="TARGET_GRAPH_X_MTP_EAGER_INTERACTION"
elif [[ "$DDECODE" == 0 && "$DEXTEND" == 1 ]]; then
  CLASS="EAGLE_DRAFT_DECODE_GRAPH_SEMANTIC_REGRESSION"
elif [[ "$DDECODE" == 1 && "$DEXTEND" == 0 ]]; then
  CLASS="EAGLE_DRAFT_EXTEND_GRAPH_SEMANTIC_REGRESSION"
elif [[ "$DDECODE" == 0 && "$DEXTEND" == 0 ]]; then
  CLASS="BOTH_SPECIALIZED_MTP_GRAPHS_INDIVIDUALLY_UNSAFE"
elif [[ "$DDECODE" == 1 && "$DEXTEND" == 1 ]]; then
  CLASS="SPECIALIZED_GRAPH_INTERACTION_ONLY"
else
  CLASS="UNCLASSIFIED"
fi

echo "MTP_GRAPH_SCOPE_CLASSIFICATION=${CLASS}"
echo "QWEN38_GITTENSOR_PP3_NVFP4_MTP_GRAPH_SCOPE_AB=PASS"
