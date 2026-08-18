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
REPEATS="${REPEATS:-2}"
SKIP_DRAFT_DECODE="${SKIP_DRAFT_DECODE:-0}"
KNOWN_DRAFT_DECODE_UNSAFE="${KNOWN_DRAFT_DECODE_UNSAFE:-0}"
ROOT="${ROOT:-/tmp/qwen38-eagle-individual-stability}"
TMP_GATES=()
SUCCESS=0
mkdir -p "$ROOT"

cleanup() {
  for f in "${TMP_GATES[@]:-}"; do
    [[ -n "$f" ]] && rm -f "$f" || true
  done
  if [[ "$SUCCESS" != "1" ]]; then
    echo '=== RESTORE GRAPH-OFF CORRECTNESS BASELINE ==='
    IMAGE="$IMAGE" MODEL="$MODEL" PORT="$PORT" \
      bash "$ROOT_DIR/qwen38_gittensor_pp3_nvfp4_256k_x2_serve.sh" || true
  fi
}
trap cleanup EXIT

make_gate() {
  local label="$1"
  local disable_draft_decode="$2"
  local disable_draft_extend="$3"
  local dst
  dst="$(mktemp "/tmp/qwen38-${label}.XXXXXX.sh")"
  TMP_GATES+=("$dst")

  python3 - \
    "$ROOT_DIR/qwen38_gittensor_pp3_nvfp4_256k_x2_gate.sh" \
    "$dst" "$disable_draft_decode" "$disable_draft_extend" <<'PY'
from pathlib import Path
import json,sys
src,dst,disable_draft_decode,disable_draft_extend=sys.argv[1:]
text=Path(src).read_text()
text=text.replace('PARTITION="${PARTITION:-22,28,14}"','PARTITION="${PARTITION:-23,28,13}"',1)
text=text.replace('CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-512}"','CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-1024}"',1)

env_anchor='  -e "PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF}" \\\n'
if text.count(env_anchor) != 1:
    raise SystemExit('eagle-individual probe: env anchor mismatch')
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
    raise SystemExit('eagle-individual probe: graph anchor mismatch')
cg={
    'decode': {'backend':'full','max_bs':2,'bs':[1,2]},
    'prefill': {'backend':'disabled'},
}
text=text.replace(
    graph_anchor,
    "    --cuda-graph-config '"+json.dumps(cg,separators=(',',':'))+"' \\\n",
    1,
)
text=text.replace(
    ' MTP ON / CUDA GRAPH OFF / SERVER WARMUP OFF',
    f' MTP ON / TARGET_VERIFY EAGER / PREFILL OFF / draft_decode_disabled={disable_draft_decode} / draft_extend_disabled={disable_draft_extend}',
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
import json,sys
path,http,rc,label=sys.argv[1],sys.argv[2],int(sys.argv[3]),sys.argv[4]
ok=rc==0 and http=='200'; content=reasoning=finish=''
try:
    d=json.load(open(path)); c=(d.get('choices') or [{}])[0]; m=c.get('message') or {}
    content=str(m.get('content') or '').strip(); reasoning=str(m.get('reasoning_content') or '').strip(); finish=str(c.get('finish_reason') or '')
    ok = ok and 'TEXT_CONTROL_OK' in ' '.join(x for x in (reasoning,content) if x)
except Exception as e:
    print(f'eagle_individual_{label}_parse_error={e!r}')
    ok=False
print(f'eagle_individual_{label}_http={http}')
print(f'eagle_individual_{label}_content={content!r}')
print(f'eagle_individual_{label}_reasoning={reasoning!r}')
print(f'eagle_individual_{label}_finish_reason={finish!r}')
print(f'eagle_individual_{label}_pass={ok}')
raise SystemExit(0 if ok else 1)
PY
}

dump_markers() {
  echo '--- ACTIVE GRAPH MARKERS ---'
  docker logs "$CONTAINER" 2>&1 | grep -E \
    'MTP-PP-GRAPH-SCOPE|MTP-PP-TARGET-VERIFY|Capture draft decode CUDA graph|Capture draft extend CUDA graph|Capture prefill CUDA graph|cuda graph' | tail -180 || true
}

classify_and_finish() {
  local class="$1"
  echo "EAGLE_INDIVIDUAL_STABILITY_CLASSIFICATION=${class}"
  dump_markers
  echo '=== RESTORE GRAPH-OFF CORRECTNESS BASELINE ==='
  IMAGE="$IMAGE" MODEL="$MODEL" PORT="$PORT" \
    bash "$ROOT_DIR/qwen38_gittensor_pp3_nvfp4_256k_x2_serve.sh" || true
  SUCCESS=1
  echo 'QWEN38_EAGLE_INDIVIDUAL_STABILITY_AB=PASS'
  exit 0
}

run_once() {
  local component="$1"
  local round="$2"
  local disable_draft_decode="$3"
  local disable_draft_extend="$4"
  local label="${component}_r${round}"
  local gate

  echo "=== ${label}: FULL 2x256K STRESS ==="
  gate="$(make_gate "$label" "$disable_draft_decode" "$disable_draft_extend")"
  PARTITION="$PARTITION" CHUNKED_PREFILL_SIZE="$CHUNKED_PREFILL_SIZE" \
  STAGE="$STAGE" DECODE_TOKENS="$DECODE_TOKENS" ROOT="$ROOT/${label}-long" \
    bash "$ROOT_DIR/qwen38_failfast_watch.sh" bash "$gate"

  echo "=== ${label}: IMMEDIATE POST-LONG SEMANTIC ==="
  semantic_control "$label"
}

echo '============================================================'
echo ' EAGLE INDIVIDUAL GRAPH STABILITY A/B: native MTP / PP3 / 2x256K'
echo ' Known facts: prefill BCG unsafe; combined EAGLE path passed once and failed once'
echo ' TARGET_VERIFY eager; prefill graph disabled for every case'
echo " repeats_per_component=${REPEATS}"
echo " skip_draft_decode=${SKIP_DRAFT_DECODE} known_draft_decode_unsafe=${KNOWN_DRAFT_DECODE_UNSAFE}"
echo '============================================================'

if (( REPEATS < 1 )); then
  echo 'ERROR: REPEATS must be >= 1' >&2
  exit 64
fi
if [[ "$SKIP_DRAFT_DECODE" == "1" && "$KNOWN_DRAFT_DECODE_UNSAFE" != "1" ]]; then
  echo 'ERROR: SKIP_DRAFT_DECODE=1 requires KNOWN_DRAFT_DECODE_UNSAFE=1' >&2
  exit 65
fi

for round in $(seq 1 "$REPEATS"); do
  if [[ "$SKIP_DRAFT_DECODE" != "1" ]]; then
    # draft-decode graph ON, draft-extend graph OFF
    if ! run_once draft_decode_only "$round" 0 1; then
      classify_and_finish "DRAFT_DECODE_CUDA_GRAPH_POST_LONG_UNSAFE_ROUND_${round}"
    fi
  else
    echo "=== draft_decode_only_r${round}: SKIPPED (known unsafe) ==="
  fi

  # draft-decode graph OFF, draft-extend graph ON
  if ! run_once draft_extend_only "$round" 1 0; then
    if [[ "$KNOWN_DRAFT_DECODE_UNSAFE" == "1" ]]; then
      classify_and_finish "BOTH_EAGLE_GRAPHS_INDIVIDUALLY_POST_LONG_UNSAFE_DRAFT_EXTEND_ROUND_${round}"
    fi
    classify_and_finish "DRAFT_EXTEND_CUDA_GRAPH_POST_LONG_UNSAFE_ROUND_${round}"
  fi
done

if [[ "$KNOWN_DRAFT_DECODE_UNSAFE" == "1" ]]; then
  classify_and_finish 'DRAFT_DECODE_UNSAFE_DRAFT_EXTEND_STABLE'
fi
classify_and_finish 'INDIVIDUAL_EAGLE_GRAPHS_STABLE_COMBINED_EAGLE_PATH_INTERMITTENT_UNSAFE'
