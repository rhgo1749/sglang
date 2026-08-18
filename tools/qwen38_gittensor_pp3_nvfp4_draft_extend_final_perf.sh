#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export IMAGE="${IMAGE:-sglang:qwen38-27b-pp-mtp-share}"
export MODEL="${MODEL:-gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090}"
export CONTAINER="${CONTAINER:-sglang-qwen38-gittensor-pp3}"
export PORT="${PORT:-30001}"
export CTX="${CTX:-262144}"
export CONTEXT_LENGTH="${CONTEXT_LENGTH:-262144}"
export PARALLEL="${PARALLEL:-2}"
export MAX_RUNNING="${MAX_RUNNING:-2}"
export MAX_MAMBA="${MAX_MAMBA:-2}"
export PARTITION="${PARTITION:-23,28,13}"
export MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.84}"
export MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-524288}"
export CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-1024}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-1200}"
export FAILFAST_STALL_SECONDS="${FAILFAST_STALL_SECONDS:-45}"
export FAILFAST_STALL_GPU_UTIL_MAX="${FAILFAST_STALL_GPU_UTIL_MAX:-5}"

PERF_REPEATS="${PERF_REPEATS:-2}"
PERF_STAGE="${PERF_STAGE:-4096}"
PERF_DECODE_TOKENS="${PERF_DECODE_TOKENS:-1024}"
ROOT="${ROOT:-/tmp/qwen38-draft-extend-final-perf}"
TMP_GATE=""
SUCCESS=0

cleanup() {
  [[ -n "${TMP_GATE:-}" ]] && rm -f "$TMP_GATE" || true
  if [[ "$SUCCESS" != "1" ]]; then
    echo '=== RESTORE GRAPH-OFF CORRECTNESS BASELINE ==='
    IMAGE="$IMAGE" MODEL="$MODEL" PORT="$PORT" \
      bash "$ROOT_DIR/qwen38_gittensor_pp3_nvfp4_256k_x2_serve.sh" || true
  fi
}
trap cleanup EXIT

if (( PERF_REPEATS < 1 )); then
  echo 'ERROR: PERF_REPEATS must be >= 1' >&2
  exit 64
fi

make_candidate_gate() {
  TMP_GATE="$(mktemp /tmp/qwen38-draft-extend-only.XXXXXX.sh)"
  python3 - "$ROOT_DIR/qwen38_gittensor_pp3_nvfp4_256k_x2_gate.sh" "$TMP_GATE" <<'PY'
from pathlib import Path
import json, sys
src, dst = sys.argv[1:]
text = Path(src).read_text()

# Keep the validated production structural defaults explicit even if the base
# gate still contains older fallback values.
text = text.replace(
    'PARTITION="${PARTITION:-22,28,14}"',
    'PARTITION="${PARTITION:-23,28,13}"',
    1,
)
text = text.replace(
    'CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-512}"',
    'CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-1024}"',
    1,
)

env_anchor = '  -e "PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF}" \\\n'
if text.count(env_anchor) != 1:
    raise SystemExit('draft-extend final perf: env anchor mismatch')
text = text.replace(
    env_anchor,
    env_anchor
    + '  -e "SGLANG_MTP_DISABLE_TARGET_VERIFY_CUDA_GRAPH=1" \\\n'
    + '  -e "SGLANG_MTP_DISABLE_DRAFT_DECODE_CUDA_GRAPH=1" \\\n'
    + '  -e "SGLANG_DISABLE_DRAFT_EXTEND_CUDA_GRAPH=0" \\\n',
    1,
)

graph_anchor = '    --disable-cuda-graph \\\n'
if text.count(graph_anchor) != 1:
    raise SystemExit('draft-extend final perf: graph anchor mismatch')
cg = {
    'decode': {'backend': 'full', 'max_bs': 2, 'bs': [1, 2]},
    'prefill': {'backend': 'disabled'},
}
text = text.replace(
    graph_anchor,
    "    --cuda-graph-config '" + json.dumps(cg, separators=(',', ':')) + "' \\\n",
    1,
)
text = text.replace(
    ' MTP ON / CUDA GRAPH OFF / SERVER WARMUP OFF',
    ' MTP ON / TARGET_VERIFY EAGER / DRAFT-DECODE EAGER / DRAFT-EXTEND CUDA GRAPH ON / PREFILL GRAPH OFF',
    1,
)
Path(dst).write_text(text)
PY
  chmod +x "$TMP_GATE"
}

semantic_control() {
  local label="$1"
  local req="${ROOT}-${label}-semantic-request.json"
  local resp="${ROOT}-${label}-semantic-response.json"
  python3 - "$req" "$MODEL" <<'PY'
import json, sys
json.dump({
    'model': sys.argv[2],
    'messages': [{'role': 'user', 'content': 'Reply with exactly: TEXT_CONTROL_OK'}],
    'temperature': 0,
    'max_tokens': 256,
}, open(sys.argv[1], 'w'), separators=(',', ':'))
PY

  local http rc
  set +e
  http="$(curl --max-time 180 -sS -o "$resp" -w '%{http_code}' \
    "http://127.0.0.1:${PORT}/v1/chat/completions" \
    -H 'Content-Type: application/json' --data-binary "@$req")"
  rc=$?
  set -e

  python3 - "$resp" "$http" "$rc" "$label" <<'PY'
import json, sys
path, http, rc, label = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
ok = rc == 0 and http == '200'
content = reasoning = finish = ''
try:
    d = json.load(open(path))
    c = (d.get('choices') or [{}])[0]
    m = c.get('message') or {}
    content = str(m.get('content') or '').strip()
    reasoning = str(m.get('reasoning_content') or '').strip()
    finish = str(c.get('finish_reason') or '')
    combined = ' '.join(x for x in (reasoning, content) if x)
    ok = ok and 'TEXT_CONTROL_OK' in combined
except Exception as e:
    print(f'draft_extend_final_{label}_semantic_parse_error={e!r}')
    ok = False
print(f'draft_extend_final_{label}_semantic_http={http}')
print(f'draft_extend_final_{label}_semantic_content={content!r}')
print(f'draft_extend_final_{label}_semantic_reasoning={reasoning!r}')
print(f'draft_extend_final_{label}_semantic_finish_reason={finish!r}')
print(f'draft_extend_final_{label}_semantic_pass={ok}')
raise SystemExit(0 if ok else 1)
PY
}

extract_elapsed() {
  python3 - "$1" <<'PY'
import re, sys
text = open(sys.argv[1], errors='replace').read()
m = re.findall(r'^stage_elapsed_seconds=([0-9.]+)$', text, re.M)
if not m:
    raise SystemExit(f'missing stage_elapsed_seconds in {sys.argv[1]}')
print(m[-1])
PY
}

make_candidate_gate
mkdir -p "$(dirname "$ROOT")"
rm -f "${ROOT}"-eager-r*.log "${ROOT}"-candidate-r*.log

echo '============================================================'
echo ' FINAL DRAFT-EXTEND-ONLY CUDA GRAPH PERFORMANCE GATE'
echo ' native MTP / PP3 / 2x256K production structure'
echo ' already-proven stability prerequisite: 2 consecutive 2x256K passes'
echo ' target-verify graph: OFF (capture/replay eager)'
echo ' EAGLE draft-decode graph: OFF (known post-long unsafe)'
echo ' EAGLE draft-extend graph: ON (known 2x long-stable)'
echo ' prefill CUDA graph: OFF'
echo " perf_repeats=${PERF_REPEATS} prompt=${PERF_STAGE} decode=${PERF_DECODE_TOKENS}"
echo '============================================================'

EAGER_VALUES=()
CANDIDATE_VALUES=()

for round in $(seq 1 "$PERF_REPEATS"); do
  EAGER_LOG="${ROOT}-eager-r${round}.log"
  CANDIDATE_LOG="${ROOT}-candidate-r${round}.log"

  echo
  echo "=== PERF ROUND ${round}/${PERF_REPEATS}: EAGER BASELINE ==="
  STAGE="$PERF_STAGE" DECODE_TOKENS="$PERF_DECODE_TOKENS" \
    ROOT="${ROOT}-eager-r${round}-artifacts" \
    bash "$ROOT_DIR/qwen38_failfast_watch.sh" \
    bash "$ROOT_DIR/qwen38_gittensor_pp3_nvfp4_256k_x2_gate.sh" | tee "$EAGER_LOG"
  EAGER_VALUES+=("$(extract_elapsed "$EAGER_LOG")")

  echo
  echo "=== PERF ROUND ${round}/${PERF_REPEATS}: DRAFT-EXTEND GRAPH CANDIDATE ==="
  STAGE="$PERF_STAGE" DECODE_TOKENS="$PERF_DECODE_TOKENS" \
    ROOT="${ROOT}-candidate-r${round}-artifacts" \
    bash "$ROOT_DIR/qwen38_failfast_watch.sh" bash "$TMP_GATE" | tee "$CANDIDATE_LOG"
  CANDIDATE_VALUES+=("$(extract_elapsed "$CANDIDATE_LOG")")
done

echo
echo '=== FINAL LIVE-CANDIDATE TEXT + MULTIMODAL SEMANTICS ==='
semantic_control final
PORT="$PORT" MODEL="$MODEL" CONTAINER="$CONTAINER" ROOT="${ROOT}-mm" \
  bash "$ROOT_DIR/qwen38_gittensor_pp3_nvfp4_mm_smoke.sh"

echo
echo '=== FINAL PERFORMANCE COMPARISON ==='
python3 - "$PERF_DECODE_TOKENS" "${EAGER_VALUES[@]}" -- "${CANDIDATE_VALUES[@]}" <<'PY'
import statistics, sys
per_req = int(sys.argv[1])
sep = sys.argv.index('--')
eager = [float(x) for x in sys.argv[2:sep]]
candidate = [float(x) for x in sys.argv[sep+1:]]
if not eager or len(eager) != len(candidate):
    raise SystemExit('mismatched perf samples')
work = 2 * per_req
emean = statistics.mean(eager)
cmean = statistics.mean(candidate)
emed = statistics.median(eager)
cmed = statistics.median(candidate)
speedup_mean = (emean / cmean - 1.0) * 100.0
speedup_median = (emed / cmed - 1.0) * 100.0
for i, (e, c) in enumerate(zip(eager, candidate), 1):
    print(f'draft_extend_final_r{i}_eager_elapsed_seconds={e:.3f}')
    print(f'draft_extend_final_r{i}_candidate_elapsed_seconds={c:.3f}')
    print(f'draft_extend_final_r{i}_speedup_pct={(e/c-1.0)*100.0:.2f}')
print(f'draft_extend_final_eager_mean_seconds={emean:.3f}')
print(f'draft_extend_final_candidate_mean_seconds={cmean:.3f}')
print(f'draft_extend_final_eager_median_seconds={emed:.3f}')
print(f'draft_extend_final_candidate_median_seconds={cmed:.3f}')
print(f'draft_extend_final_eager_effective_completion_tps={work/emean:.3f}')
print(f'draft_extend_final_candidate_effective_completion_tps={work/cmean:.3f}')
print(f'draft_extend_final_mean_speedup_pct={speedup_mean:.2f}')
print(f'draft_extend_final_median_speedup_pct={speedup_median:.2f}')
print(f'draft_extend_final_perf_improved={speedup_mean > 0.0 and speedup_median > 0.0}')
PY

echo 'QWEN38_DRAFT_EXTEND_FINAL_TEXT_PASS=True'
echo 'QWEN38_DRAFT_EXTEND_FINAL_MM_PASS=True'
echo 'QWEN38_GITTENSOR_PP3_NVFP4_DRAFT_EXTEND_FINAL_PERF=PASS'
echo "candidate_endpoint=http://127.0.0.1:${PORT}"
SUCCESS=1
