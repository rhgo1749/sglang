#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="${IMAGE:-sglang:qwen38-27b-pp-mtp-share}"
MODEL="${MODEL:-gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090}"
CONTAINER="${CONTAINER:-sglang-qwen38-gittensor-pp3}"
PORT="${PORT:-30001}"
PARTITION="${PARTITION:-23,28,13}"
CTX="${CTX:-262144}"
MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-524288}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.84}"
CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-1024}"
ROOT="${ROOT:-/tmp/qwen38-post-long-state-ab}"
TMP_GATE=""
SUCCESS=0
mkdir -p "$ROOT"

CG_JSON='{"decode":{"backend":"full","max_bs":2,"bs":[1,2]},"prefill":{"backend":"breakable","bs":[512,1024]}}'

cleanup() {
  [[ -n "${TMP_GATE:-}" ]] && rm -f "$TMP_GATE" || true
  if [[ "$SUCCESS" != "1" ]]; then
    echo "=== RESTORE CORRECTNESS BASELINE MTP SERVER ==="
    IMAGE="$IMAGE" MODEL="$MODEL" PORT="$PORT" \
      bash "$ROOT_DIR/qwen38_gittensor_pp3_nvfp4_256k_x2_serve.sh" || true
  fi
}
trap cleanup EXIT

make_hybrid_gate() {
  TMP_GATE="$(mktemp /tmp/qwen38-post-long-hybrid-gate.XXXXXX.sh)"
  python3 - "$ROOT_DIR/qwen38_gittensor_pp3_nvfp4_256k_x2_gate.sh" "$TMP_GATE" <<'PY'
from pathlib import Path
import sys
src, dst = map(str, sys.argv[1:])
text = Path(src).read_text()
text = text.replace('PARTITION="${PARTITION:-22,28,14}"','PARTITION="${PARTITION:-23,28,13}"',1)
text = text.replace('CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-512}"','CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-1024}"',1)
env_anchor='  -e "PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF}" \\\n'
if text.count(env_anchor) != 1:
    raise SystemExit('post-long probe: env anchor mismatch')
text=text.replace(env_anchor, env_anchor
    +'  -e "SGLANG_MTP_DISABLE_TARGET_VERIFY_CUDA_GRAPH=1" \\\n'
    +'  -e "SGLANG_MTP_DISABLE_DRAFT_DECODE_CUDA_GRAPH=0" \\\n'
    +'  -e "SGLANG_DISABLE_DRAFT_EXTEND_CUDA_GRAPH=0" \\\n',1)
graph_anchor='    --disable-cuda-graph \\\n'
if text.count(graph_anchor) != 1:
    raise SystemExit('post-long probe: graph anchor mismatch')
text=text.replace(graph_anchor,
    "    --cuda-graph-config '{\"decode\":{\"backend\":\"full\",\"max_bs\":2,\"bs\":[1,2]},\"prefill\":{\"backend\":\"breakable\",\"bs\":[512,1024]}}' \\\n",1)
text=text.replace(' MTP ON / CUDA GRAPH OFF / SERVER WARMUP OFF',' MTP ON / TARGET_VERIFY EAGER / EAGLE GRAPHS ON / PP PREFILL BCG ON',1)
Path(dst).write_text(text)
PY
  chmod +x "$TMP_GATE"
}

semantic_control() {
  local label="$1"
  local req="$ROOT/${label}-req.json"
  local resp="$ROOT/${label}-resp.json"
  python3 - "$req" "$MODEL" <<'PY'
import json,sys
json.dump({
  "model":sys.argv[2],
  "messages":[{"role":"user","content":"Reply with exactly: TEXT_CONTROL_OK"}],
  "temperature":0,
  "max_tokens":256,
},open(sys.argv[1],"w"),separators=(",",":"))
PY
  local http rc
  set +e
  http="$(curl --max-time 180 -sS -o "$resp" -w '%{http_code}' \
    "http://127.0.0.1:${PORT}/v1/chat/completions" \
    -H 'Content-Type: application/json' --data-binary "@$req")"
  rc=$?
  set -e
  python3 - "$resp" "$http" "$rc" "$label" <<'PY'
import json,pathlib,sys
path,http,rc,label=sys.argv[1],sys.argv[2],int(sys.argv[3]),sys.argv[4]
ok=rc==0 and http=='200'; content=reasoning=finish=''
try:
    d=json.load(open(path)); c=(d.get('choices') or [{}])[0]; m=c.get('message') or {}
    content=str(m.get('content') or '').strip(); reasoning=str(m.get('reasoning_content') or '').strip(); finish=str(c.get('finish_reason') or '')
    ok = ok and 'TEXT_CONTROL_OK' in (' '.join(x for x in (reasoning,content) if x))
except Exception as e:
    print(f'post_long_{label}_parse_error={e!r}'); ok=False
print(f'post_long_{label}_http={http}')
print(f'post_long_{label}_content={content!r}')
print(f'post_long_{label}_reasoning={reasoning!r}')
print(f'post_long_{label}_finish_reason={finish!r}')
print(f'post_long_{label}_pass={ok}')
pathlib.Path(path+'.pass').write_text('1' if ok else '0')
PY
  cat "$resp.pass"
}

launch_hybrid_only() {
  docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
  docker run -d \
    --name "$CONTAINER" \
    --gpus '"device=0,2,1"' \
    --ipc=host --shm-size 32g \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    -e "SGLANG_PP_LAYER_PARTITION=${PARTITION}" \
    -e "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" \
    -e "SGLANG_MTP_DISABLE_TARGET_VERIFY_CUDA_GRAPH=1" \
    -e "SGLANG_MTP_DISABLE_DRAFT_DECODE_CUDA_GRAPH=0" \
    -e "SGLANG_DISABLE_DRAFT_EXTEND_CUDA_GRAPH=0" \
    -p "${PORT}:30000" \
    -v sglang-hf-cache:/root/.cache/huggingface \
    "$IMAGE" python3 -m sglang.launch_server \
      --model-path "$MODEL" --tp-size 1 --pp-size 3 --trust-remote-code \
      --context-length "$CTX" --max-total-tokens "$MAX_TOTAL_TOKENS" \
      --speculative-algorithm NEXTN --speculative-num-steps 3 \
      --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
      --skip-server-warmup --kv-cache-dtype nvfp4 \
      --attention-backend flashinfer --prefill-attention-backend flashinfer \
      --decode-attention-backend trtllm_mha --page-size 64 \
      --mem-fraction-static "$MEM_FRACTION_STATIC" --max-running-requests 2 \
      --max-mamba-cache-size 2 --mamba-ssm-dtype bfloat16 \
      --mamba-radix-cache-strategy extra_buffer_lazy --disable-radix-cache \
      --mm-feature-transport cpu --chunked-prefill-size "$CHUNKED_PREFILL_SIZE" \
      --cuda-graph-config "$CG_JSON" --disable-flashinfer-autotune \
      --reasoning-parser qwen3 --tool-call-parser qwen3_coder \
      --host 0.0.0.0 --port 30000 >/dev/null
  timeout 300 bash -c '
    while true; do
      curl -fsS http://127.0.0.1:'"$PORT"'/model_info >/dev/null 2>&1 && exit 0
      [[ "$(docker inspect -f "{{.State.Running}}" '"$CONTAINER"' 2>/dev/null || echo false)" == true ]] || exit 2
      sleep 1
    done
  '
}

error_tail() {
  echo '--- FIRST ROOT ERROR CANDIDATES ---'
  docker logs "$CONTAINER" 2>&1 | grep -Ei 'Traceback|Scheduler hit an exception|RuntimeError|AssertionError|ValueError|CUDA error|out of memory|MTP-PP' | head -160 || true
}

make_hybrid_gate

echo '============================================================'
echo ' POST-LONG STATE A/B: native MTP / PP3 / 2x256K'
echo '============================================================'

echo '=== A: HYBRID FULL LONG STRESS ==='
PARTITION="$PARTITION" CHUNKED_PREFILL_SIZE="$CHUNKED_PREFILL_SIZE" \
STAGE=262000 DECODE_TOKENS=8 ROOT="$ROOT/hybrid-long" \
  bash "$ROOT_DIR/qwen38_failfast_watch.sh" bash "$TMP_GATE"

echo '=== A1: IMMEDIATE POST-LONG SEMANTIC ==='
A_IMMEDIATE="$(semantic_control hybrid_immediate | tee /dev/stderr | tail -n1)"

echo '=== A2: SETTLED POST-LONG SEMANTIC ==='
A_SETTLED="$A_IMMEDIATE"
for delay in 2 8 20; do
  if [[ "$A_SETTLED" == 1 ]]; then break; fi
  echo "post_long_wait_seconds=${delay}"
  sleep "$delay"
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits || true
  A_SETTLED="$(semantic_control "hybrid_after_${delay}s" | tee /dev/stderr | tail -n1)"
done

if [[ "$A_IMMEDIATE" == 0 && "$A_SETTLED" == 1 ]]; then
  CLASS='POST_RESPONSE_TAIL_SYNC_WINDOW'
  echo "POST_LONG_STATE_CLASSIFICATION=${CLASS}"
  error_tail
  exit 0
fi

if [[ "$A_SETTLED" == 1 ]]; then
  CLASS='NO_REPRO_OR_IMMEDIATE_PASS'
  echo "POST_LONG_STATE_CLASSIFICATION=${CLASS}"
  SUCCESS=1
  exit 0
fi

echo '=== B: FRESH HYBRID RESTART CONTROL ==='
launch_hybrid_only
B_FRESH="$(semantic_control hybrid_fresh_restart | tee /dev/stderr | tail -n1)"
if [[ "$B_FRESH" != 1 ]]; then
  echo 'POST_LONG_STATE_CLASSIFICATION=HYBRID_FRESH_SERVER_SEMANTIC_REGRESSION'
  error_tail
  exit 1
fi

echo '=== C: EAGER BASELINE FULL LONG STRESS ==='
PARTITION="$PARTITION" CHUNKED_PREFILL_SIZE="$CHUNKED_PREFILL_SIZE" \
STAGE=262000 DECODE_TOKENS=8 ROOT="$ROOT/eager-long" \
  bash "$ROOT_DIR/qwen38_failfast_watch.sh" \
  bash "$ROOT_DIR/qwen38_gittensor_pp3_nvfp4_256k_x2_gate.sh"
C_IMMEDIATE="$(semantic_control eager_immediate | tee /dev/stderr | tail -n1)"
if [[ "$C_IMMEDIATE" != 1 ]]; then
  sleep 20
  C_SETTLED="$(semantic_control eager_after_20s | tee /dev/stderr | tail -n1)"
else
  C_SETTLED=1
fi

if [[ "$C_SETTLED" == 1 ]]; then
  CLASS='HYBRID_GRAPH_POST_LONG_STATE_CORRUPTION'
else
  CLASS='GENERIC_NATIVE_MTP_POST_LONG_STATE_CORRUPTION'
fi

echo "POST_LONG_STATE_CLASSIFICATION=${CLASS}"
error_tail
# Leave the known-correct eager baseline running after a diagnostic failure.
IMAGE="$IMAGE" MODEL="$MODEL" PORT="$PORT" \
  bash "$ROOT_DIR/qwen38_gittensor_pp3_nvfp4_256k_x2_serve.sh" || true
SUCCESS=1
echo 'QWEN38_GITTENSOR_PP3_NVFP4_POST_LONG_STATE_AB=PASS'
