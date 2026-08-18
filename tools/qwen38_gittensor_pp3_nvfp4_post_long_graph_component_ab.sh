#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="${IMAGE:-sglang:qwen38-27b-pp-mtp-share}"
MODEL="${MODEL:-gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090}"
CONTAINER="${CONTAINER:-sglang-qwen38-gittensor-pp3}"
PORT="${PORT:-30001}"
PARTITION="${PARTITION:-23,28,13}"
CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-1024}"
STAGE="${STAGE:-262000}"
DECODE_TOKENS="${DECODE_TOKENS:-8}"
ROOT="${ROOT:-/tmp/qwen38-post-long-graph-components}"
TMP_GATES=()
SUCCESS=0
mkdir -p "$ROOT"

cleanup() {
  for f in "${TMP_GATES[@]:-}"; do
    [[ -n "$f" ]] && rm -f "$f" || true
  done
  if [[ "$SUCCESS" != "1" ]]; then
    echo '=== RESTORE CORRECTNESS BASELINE MTP SERVER ==='
    IMAGE="$IMAGE" MODEL="$MODEL" PORT="$PORT" \
      bash "$ROOT_DIR/qwen38_gittensor_pp3_nvfp4_256k_x2_serve.sh" || true
  fi
}
trap cleanup EXIT

make_gate() {
  local label="$1"
  local decode_backend="$2"      # full | disabled
  local prefill_backend="$3"     # breakable | disabled
  local disable_draft_decode="$4" # 0 | 1
  local disable_draft_extend="$5" # 0 | 1
  local dst
  dst="$(mktemp "/tmp/qwen38-${label}.XXXXXX.sh")"
  TMP_GATES+=("$dst")

  python3 - \
    "$ROOT_DIR/qwen38_gittensor_pp3_nvfp4_256k_x2_gate.sh" \
    "$dst" "$decode_backend" "$prefill_backend" \
    "$disable_draft_decode" "$disable_draft_extend" <<'PY'
from pathlib import Path
import json,sys
src,dst,decode_backend,prefill_backend,disable_draft_decode,disable_draft_extend=sys.argv[1:]
text=Path(src).read_text()
text=text.replace('PARTITION="${PARTITION:-22,28,14}"','PARTITION="${PARTITION:-23,28,13}"',1)
text=text.replace('CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-512}"','CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-1024}"',1)

env_anchor='  -e "PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF}" \\\n'
if text.count(env_anchor) != 1:
    raise SystemExit('graph-component probe: env anchor mismatch')
text=text.replace(
    env_anchor,
    env_anchor
    +'  -e "SGLANG_MTP_DISABLE_TARGET_VERIFY_CUDA_GRAPH=1" \\\n'
    +f'  -e "SGLANG_MTP_DISABLE_DRAFT_DECODE_CUDA_GRAPH={disable_draft_decode}" \\\n'
    +f'  -e "SGLANG_DISABLE_DRAFT_EXTEND_CUDA_GRAPH={disable_draft_extend}" \\\n',
    1,
)

graph_anchor='    --disable-cuda-graph \\\n'
if text.count(graph_anchor) != 1:
    raise SystemExit('graph-component probe: graph anchor mismatch')
cg={
    'decode': {'backend': decode_backend, 'max_bs': 2, 'bs': [1,2]},
    'prefill': (
        {'backend':'breakable','bs':[512,1024]}
        if prefill_backend == 'breakable'
        else {'backend':'disabled'}
    ),
}
text=text.replace(
    graph_anchor,
    "    --cuda-graph-config '"+json.dumps(cg,separators=(',',':'))+"' \\\n",
    1,
)
text=text.replace(
    ' MTP ON / CUDA GRAPH OFF / SERVER WARMUP OFF',
    f' MTP ON / TARGET_VERIFY EAGER / decode={decode_backend} / prefill={prefill_backend} / draft_decode_disabled={disable_draft_decode} / draft_extend_disabled={disable_draft_extend}',
    1,
)
Path(dst).write_text(text)
PY
  chmod +x "$dst"
  printf '%s\n' "$dst"
}

semantic_control() {
  local label="$1"
  local req="$ROOT/${label}-req.json"
  local resp="$ROOT/${label}-resp.json"
  python3 - "$req" "$MODEL" <<'PY'
import json,sys
json.dump({
  'model':sys.argv[2],
  'messages':[{'role':'user','content':'Reply with exactly: TEXT_CONTROL_OK'}],
  'temperature':0,
  'max_tokens':256,
},open(sys.argv[1],'w'),separators=(',',':'))
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
    print(f'graph_component_{label}_parse_error={e!r}'); ok=False
print(f'graph_component_{label}_http={http}')
print(f'graph_component_{label}_content={content!r}')
print(f'graph_component_{label}_reasoning={reasoning!r}')
print(f'graph_component_{label}_finish_reason={finish!r}')
print(f'graph_component_{label}_pass={ok}')
pathlib.Path(path+'.pass').write_text('1' if ok else '0')
PY
  cat "$resp.pass"
}

run_variant() {
  local label="$1"
  local decode_backend="$2"
  local prefill_backend="$3"
  local disable_draft_decode="$4"
  local disable_draft_extend="$5"
  local gate result

  echo "=== ${label}: FULL 2x256K STRESS ==="
  gate="$(make_gate "$label" "$decode_backend" "$prefill_backend" "$disable_draft_decode" "$disable_draft_extend")"
  PARTITION="$PARTITION" CHUNKED_PREFILL_SIZE="$CHUNKED_PREFILL_SIZE" \
  STAGE="$STAGE" DECODE_TOKENS="$DECODE_TOKENS" ROOT="$ROOT/${label}-long" \
    bash "$ROOT_DIR/qwen38_failfast_watch.sh" bash "$gate"

  echo "=== ${label}: IMMEDIATE POST-LONG SEMANTIC ==="
  result="$(semantic_control "$label" | tee /dev/stderr | tail -n1)"
  printf '%s\n' "$result"
}

classify_and_finish() {
  local class="$1"
  echo "POST_LONG_GRAPH_COMPONENT_CLASSIFICATION=${class}"
  echo '--- ACTIVE GRAPH MARKERS ---'
  docker logs "$CONTAINER" 2>&1 | grep -E \
    'MTP-PP-GRAPH-SCOPE|MTP-PP-TARGET-VERIFY|Capture draft decode CUDA graph|Capture draft extend CUDA graph|Capture prefill CUDA graph|cuda graph' | tail -160 || true
  IMAGE="$IMAGE" MODEL="$MODEL" PORT="$PORT" \
    bash "$ROOT_DIR/qwen38_gittensor_pp3_nvfp4_256k_x2_serve.sh" || true
  SUCCESS=1
  echo 'QWEN38_POST_LONG_GRAPH_COMPONENT_AB=PASS'
  exit 0
}

echo '============================================================'
echo ' POST-LONG GRAPH COMPONENT A/B: native MTP / PP3 / 2x256K'
echo ' Known baseline: all hybrid graphs ON corrupts; all graphs OFF passes'
echo '============================================================'

# 1) Clean prefill-graph-only test. decode backend disabled means the EAGLE
# specialized decode/draft-extend capture routine returns before either graph,
# while breakable prefill graphs remain enabled. TARGET_VERIFY is eager.
PREFILL_ONLY="$(run_variant prefill_only disabled breakable 1 1 | tee /dev/stderr | tail -n1)"
if [[ "$PREFILL_ONLY" != 1 ]]; then
  classify_and_finish 'PREFILL_BCG_POST_LONG_STATE_CORRUPTION'
fi

# 2) EAGLE graph family with prefill graphs disabled. TARGET_VERIFY remains eager.
EAGLE_BOTH="$(run_variant eagle_both full disabled 0 0 | tee /dev/stderr | tail -n1)"
if [[ "$EAGLE_BOTH" == 1 ]]; then
  # Each family is clean in isolation, but the already-proven all-on hybrid is not.
  classify_and_finish 'PREFILL_X_EAGLE_GRAPH_INTERACTION'
fi

# 3) Split the EAGLE family. Keep prefill disabled so each specialized graph is
# tested without the BCG family in the process.
DRAFT_DECODE="$(run_variant draft_decode_only full disabled 0 1 | tee /dev/stderr | tail -n1)"
if [[ "$DRAFT_DECODE" != 1 ]]; then
  classify_and_finish 'DRAFT_DECODE_CUDA_GRAPH_POST_LONG_STATE_CORRUPTION'
fi

DRAFT_EXTEND="$(run_variant draft_extend_only full disabled 1 0 | tee /dev/stderr | tail -n1)"
if [[ "$DRAFT_EXTEND" != 1 ]]; then
  classify_and_finish 'DRAFT_EXTEND_CUDA_GRAPH_POST_LONG_STATE_CORRUPTION'
fi

# Both individual specialized graphs pass, but together failed in step 2.
classify_and_finish 'DRAFT_DECODE_X_DRAFT_EXTEND_GRAPH_INTERACTION'
