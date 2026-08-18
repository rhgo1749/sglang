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
ROOT="${ROOT:-/tmp/qwen38-mtp-hybrid-graph-ab}"
RESTORE_MTP="${RESTORE_MTP:-1}"
mkdir -p "$ROOT"

CG_PREFILL_OFF='{"decode":{"backend":"full","max_bs":2,"bs":[1,2]},"prefill":{"backend":"disabled"}}'
CG_PREFILL_BCG='{"decode":{"backend":"full","max_bs":2,"bs":[1,2]},"prefill":{"backend":"breakable","bs":[512,1024]}}'

restore_mtp() {
  docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
  if [[ "$RESTORE_MTP" == "1" ]]; then
    echo "=== RESTORE CORRECTNESS BASELINE MTP SERVER ==="
    IMAGE="$IMAGE" MODEL="$MODEL" PORT="$PORT" \
      bash tools/qwen38_gittensor_pp3_nvfp4_256k_x2_serve.sh || true
  fi
}
trap restore_mtp EXIT

first_errors() {
  local mode="$1"
  echo "--- ${mode}: FIRST ROOT ERROR CANDIDATES ---"
  docker logs "$CONTAINER" 2>&1 | \
    grep -Ei 'Traceback|Scheduler hit an exception|RuntimeError|AssertionError|ValueError|AcceleratorError|CUDA error|out of memory|Connection closed|NCCL|Gloo|MTP-PP' | \
    head -120 || true
  echo "--- ${mode}: SERVER TAIL ---"
  docker logs "$CONTAINER" 2>&1 | tail -120 || true
}

launch_mode() {
  local mode="$1"
  local draft_decode_disable="$2"
  local draft_extend_disable="$3"
  local prefill_mode="$4"
  local cg_json="$CG_PREFILL_OFF"
  if [[ "$prefill_mode" == "bcg" ]]; then
    cg_json="$CG_PREFILL_BCG"
  fi

  docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
  echo "=== ${mode} ==="
  echo "  target_verify_capture=disabled"
  echo "  target_verify_replay=eager"
  echo "  draft_decode_graph=$([[ "$draft_decode_disable" == 0 ]] && echo on || echo off)"
  echo "  draft_extend_graph=$([[ "$draft_extend_disable" == 0 ]] && echo on || echo off)"
  echo "  pp_prefill_graph=${prefill_mode}"

  docker run -d \
    --name "$CONTAINER" \
    --gpus '"device=0,2,1"' \
    --ipc=host \
    --shm-size 32g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -e "SGLANG_PP_LAYER_PARTITION=${PARTITION}" \
    -e "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" \
    -e "SGLANG_MTP_DISABLE_TARGET_VERIFY_CUDA_GRAPH=1" \
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
      --cuda-graph-config "$cg_json" \
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
    echo "mtp_hybrid_${mode}_boot_pass=False"
    first_errors "$mode"
    printf '0' >"$ROOT/${mode}.pass"
    return 1
  fi
  echo "mtp_hybrid_${mode}_boot_pass=True"
  docker logs "$CONTAINER" 2>&1 | \
    grep -E 'MTP-PP-TARGET-VERIFY-(GRAPH|CAPTURE)|MTP-PP-GRAPH-SCOPE|Capture (target prefill|draft decode|draft extend) CUDA graph' | \
    tail -100 || true
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

  local http rc running
  set +e
  http="$(curl --max-time 180 -sS -o "$resp" -w '%{http_code}' \
    "http://127.0.0.1:${PORT}/v1/chat/completions" \
    -H 'Content-Type: application/json' --data-binary "@$req")"
  rc=$?
  set -e
  running="$(docker inspect -f '{{.State.Running}}' "$CONTAINER" 2>/dev/null || echo false)"
  echo "mtp_hybrid_${mode}_curl_rc=${rc}"
  echo "mtp_hybrid_${mode}_http=${http}"
  echo "mtp_hybrid_${mode}_server_running_after_request=${running}"

  set +e
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
    print(f"mtp_hybrid_{mode}_content={content!r}")
    print(f"mtp_hybrid_{mode}_reasoning={reasoning!r}")
    print(f"mtp_hybrid_{mode}_finish_reason={finish!r}")
    print(f"mtp_hybrid_{mode}_semantic_pass={semantic}")
except Exception as e:
    print(f"mtp_hybrid_{mode}_parse_error={e!r}")
    ok = False
print(f"mtp_hybrid_{mode}_pass={ok}")
pathlib.Path(path + ".pass").write_text("1" if ok else "0")
raise SystemExit(0 if ok else 1)
PY
  local verify_rc=$?
  set -e
  cp "$resp.pass" "$ROOT/${mode}.pass"
  if (( verify_rc != 0 )); then
    first_errors "$mode"
    echo "--- ${mode}: RAW RESPONSE ---"
    cat "$resp" 2>/dev/null || true
    echo
  fi
}

run_mode() {
  local mode="$1" dd="$2" de="$3" pf="$4"
  if launch_mode "$mode" "$dd" "$de" "$pf"; then
    semantic_control "$mode"
  fi
}

# 1) Does skipping TARGET_VERIFY *capture*, not only replay, recover semantics?
run_mode verify_capture_off_all_draft_eager 1 1 off
# 2-4) Re-enable each EAGLE graph without a target-verify graph contaminating it.
run_mode verify_capture_off_draft_decode 0 1 off
run_mode verify_capture_off_draft_extend 1 0 off
run_mode verify_capture_off_both_draft 0 0 off
# 5) PP-aware prefill BCG by itself. Decode/verify stays eager.
run_mode verify_capture_off_prefill_bcg 1 1 bcg
# 6) Maximum hybrid candidate: PP prefill BCG + both EAGLE graphs, verify eager.
run_mode verify_capture_off_both_draft_prefill_bcg 0 0 bcg

read_pass() { [[ -f "$ROOT/$1.pass" ]] && cat "$ROOT/$1.pass" || echo 0; }
BASE="$(read_pass verify_capture_off_all_draft_eager)"
DDECODE="$(read_pass verify_capture_off_draft_decode)"
DEXTEND="$(read_pass verify_capture_off_draft_extend)"
DBOTH="$(read_pass verify_capture_off_both_draft)"
PREFILL="$(read_pass verify_capture_off_prefill_bcg)"
HYBRID="$(read_pass verify_capture_off_both_draft_prefill_bcg)"

echo "mtp_hybrid_verify_capture_off_all_draft_eager_pass=$([[ "$BASE" == 1 ]] && echo True || echo False)"
echo "mtp_hybrid_draft_decode_pass=$([[ "$DDECODE" == 1 ]] && echo True || echo False)"
echo "mtp_hybrid_draft_extend_pass=$([[ "$DEXTEND" == 1 ]] && echo True || echo False)"
echo "mtp_hybrid_both_draft_pass=$([[ "$DBOTH" == 1 ]] && echo True || echo False)"
echo "mtp_hybrid_pp_prefill_bcg_pass=$([[ "$PREFILL" == 1 ]] && echo True || echo False)"
echo "mtp_hybrid_max_candidate_pass=$([[ "$HYBRID" == 1 ]] && echo True || echo False)"

if [[ "$BASE" != 1 ]]; then
  CLASS="TARGET_VERIFY_CAPTURE_SKIP_INSUFFICIENT"
elif [[ "$DDECODE" == 0 && "$DEXTEND" == 1 ]]; then
  CLASS="EAGLE_DRAFT_DECODE_GRAPH_UNSAFE_AFTER_VERIFY_FIX"
elif [[ "$DDECODE" == 1 && "$DEXTEND" == 0 ]]; then
  CLASS="EAGLE_DRAFT_EXTEND_GRAPH_UNSAFE_AFTER_VERIFY_FIX"
elif [[ "$DDECODE" == 0 && "$DEXTEND" == 0 ]]; then
  CLASS="BOTH_EAGLE_GRAPHS_UNSAFE_AFTER_VERIFY_FIX"
elif [[ "$DBOTH" != 1 && "$DDECODE" == 1 && "$DEXTEND" == 1 ]]; then
  CLASS="EAGLE_GRAPH_INTERACTION_ONLY"
elif [[ "$PREFILL" != 1 ]]; then
  CLASS="PP_PREFILL_GRAPH_SEMANTIC_REGRESSION"
elif [[ "$HYBRID" != 1 ]]; then
  CLASS="PP_PREFILL_X_EAGLE_GRAPH_INTERACTION"
else
  CLASS="HYBRID_PP_PREFILL_AND_EAGLE_GRAPHS_SEMANTIC_PASS"
fi

echo "MTP_HYBRID_GRAPH_CLASSIFICATION=${CLASS}"

# Only when the maximum text candidate is semantically green, exercise the
# sparse multimodal bridge under the same PP-prefill/EAGLE graph combination.
MM_OK=NotRun
if [[ "$HYBRID" == 1 ]]; then
  set +e
  PORT="$PORT" CONTAINER="$CONTAINER" ROOT="$ROOT/mm" \
    bash tools/qwen38_gittensor_pp3_nvfp4_mm_smoke.sh
  mm_rc=$?
  set -e
  MM_OK=$([[ "$mm_rc" == 0 ]] && echo True || echo False)
fi
echo "mtp_hybrid_max_candidate_mm_pass=${MM_OK}"
echo "QWEN38_GITTENSOR_PP3_NVFP4_MTP_HYBRID_GRAPH_AB=PASS"
