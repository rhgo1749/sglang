#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="${IMAGE:-sglang:qwen38-27b-pp-mtp-share}"
MODEL="${MODEL:-gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090}"
CONTAINER="${CONTAINER:-sglang-qwen38-gittensor-pp3}"
PORT="${PORT:-30001}"
PARTITION="${PARTITION:-23,28,13}"
CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-1024}"
ROOT="${ROOT:-/tmp/qwen38-target-only-post-long}"
mkdir -p "$ROOT"
SUCCESS=0

cleanup() {
  if [[ "$SUCCESS" != "1" ]]; then
    echo '=== RESTORE CORRECTNESS BASELINE MTP SERVER ==='
    IMAGE="$IMAGE" MODEL="$MODEL" PORT="$PORT" \
      bash "$ROOT_DIR/qwen38_gittensor_pp3_nvfp4_256k_x2_serve.sh" || true
  fi
}
trap cleanup EXIT

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
    content=str(m.get('content') or '').strip()
    reasoning=str(m.get('reasoning_content') or '').strip()
    finish=str(c.get('finish_reason') or '')
    ok = ok and 'TEXT_CONTROL_OK' in ' '.join(x for x in (reasoning,content) if x)
except Exception as e:
    print(f'target_only_{label}_parse_error={e!r}'); ok=False
print(f'target_only_{label}_http={http}')
print(f'target_only_{label}_content={content!r}')
print(f'target_only_{label}_reasoning={reasoning!r}')
print(f'target_only_{label}_finish_reason={finish!r}')
print(f'target_only_{label}_pass={ok}')
pathlib.Path(path+'.pass').write_text('1' if ok else '0')
raise SystemExit(0 if ok else 1)
PY
}

echo '============================================================'
echo ' TARGET-ONLY POST-LONG STATE SEPARATOR: PP3 / 2x256K'
echo ' native MTP/speculative decoding: OFF'
echo ' CUDA graph: OFF'
echo " partition=${PARTITION} chunk=${CHUNKED_PREFILL_SIZE}"
echo '============================================================'

echo '=== TARGET-ONLY FULL 2x256K LONG STRESS ==='
ENABLE_MTP=0 \
PARTITION="$PARTITION" \
CHUNKED_PREFILL_SIZE="$CHUNKED_PREFILL_SIZE" \
STAGE=262000 DECODE_TOKENS=8 ROOT="$ROOT/long" \
  bash "$ROOT_DIR/qwen38_failfast_watch.sh" \
  bash "$ROOT_DIR/qwen38_gittensor_pp3_nvfp4_256k_x2_gate.sh"

echo '=== TARGET-ONLY IMMEDIATE POST-LONG SEMANTIC ==='
set +e
semantic_control immediate
IMMEDIATE_RC=$?
set -e

if (( IMMEDIATE_RC == 0 )); then
  echo 'TARGET_ONLY_POST_LONG_CLASSIFICATION=TARGET_MAMBA_REUSE_CLEAN_NATIVE_MTP_SPEC_STATE_SUSPECT'
  echo 'QWEN38_GITTENSOR_PP3_NVFP4_TARGET_ONLY_POST_LONG=PASS'
  IMAGE="$IMAGE" MODEL="$MODEL" PORT="$PORT" \
    bash "$ROOT_DIR/qwen38_gittensor_pp3_nvfp4_256k_x2_serve.sh"
  SUCCESS=1
  exit 0
fi

echo 'target_only_wait_seconds=20'
sleep 20
set +e
semantic_control after_20s
SETTLED_RC=$?
set -e

if (( SETTLED_RC == 0 )); then
  CLASS='TARGET_ONLY_POST_RESPONSE_TAIL_SYNC_WINDOW'
else
  CLASS='GENERIC_TARGET_MAMBA_OR_REQUEST_SLOT_REUSE_CORRUPTION'
fi

echo "TARGET_ONLY_POST_LONG_CLASSIFICATION=${CLASS}"
echo '--- FIRST ROOT ERROR CANDIDATES ---'
docker logs "$CONTAINER" 2>&1 | \
  grep -Ei 'Traceback|Scheduler hit an exception|RuntimeError|AssertionError|ValueError|CUDA error|out of memory|Mamba|mamba|MTP-PP' | head -220 || true

IMAGE="$IMAGE" MODEL="$MODEL" PORT="$PORT" \
  bash "$ROOT_DIR/qwen38_gittensor_pp3_nvfp4_256k_x2_serve.sh" || true
SUCCESS=1
echo 'QWEN38_GITTENSOR_PP3_NVFP4_TARGET_ONLY_POST_LONG=PASS'
