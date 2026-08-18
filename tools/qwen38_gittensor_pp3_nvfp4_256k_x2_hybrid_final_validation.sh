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

ROOT="${ROOT:-/tmp/qwen38-x2-eagle-final}"
TMP_GATE=""
SUCCESS=0

cleanup() {
  [[ -n "${TMP_GATE:-}" ]] && rm -f "$TMP_GATE" || true
  if [[ "$SUCCESS" != "1" ]]; then
    echo "=== RESTORE PRODUCTION EAGLE-GRAPH SERVER ==="
    IMAGE="$IMAGE" MODEL="$MODEL" PORT="$PORT" \
      bash "$ROOT_DIR/qwen38_gittensor_pp3_nvfp4_256k_x2_serve.sh" || true
  fi
}
trap cleanup EXIT

make_candidate_gate() {
  TMP_GATE="$(mktemp /tmp/qwen38-x2-eagle-gate.XXXXXX.sh)"
  python3 - "$ROOT_DIR/qwen38_gittensor_pp3_nvfp4_256k_x2_gate.sh" "$TMP_GATE" <<'PY'
from pathlib import Path
import sys
src, dst = map(str, sys.argv[1:])
text = Path(src).read_text()

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
env_insert = (
    env_anchor
    + '  -e "SGLANG_MTP_DISABLE_TARGET_VERIFY_CUDA_GRAPH=1" \\\n'
    + '  -e "SGLANG_MTP_DISABLE_DRAFT_DECODE_CUDA_GRAPH=0" \\\n'
    + '  -e "SGLANG_DISABLE_DRAFT_EXTEND_CUDA_GRAPH=0" \\\n'
)
if text.count(env_anchor) != 1:
    raise SystemExit('candidate gate: PYTORCH_CUDA_ALLOC_CONF anchor mismatch')
text = text.replace(env_anchor, env_insert, 1)

graph_anchor = '    --disable-cuda-graph \\\n'
graph_repl = (
    "    --cuda-graph-config "
    "'{\"decode\":{\"backend\":\"full\",\"max_bs\":2,\"bs\":[1,2]},"
    "\"prefill\":{\"backend\":\"disabled\"}}' \\\n"
)
if text.count(graph_anchor) != 1:
    raise SystemExit('candidate gate: --disable-cuda-graph anchor mismatch')
text = text.replace(graph_anchor, graph_repl, 1)
text = text.replace(
    ' MTP ON / CUDA GRAPH OFF / SERVER WARMUP OFF',
    ' MTP ON / TARGET_VERIFY EAGER / EAGLE GRAPHS ON / PREFILL GRAPH OFF',
    1,
)
Path(dst).write_text(text)
PY
  chmod +x "$TMP_GATE"
}

semantic_control() {
  local label="$1"
  local req="$ROOT-${label}-semantic-request.json"
  local resp="$ROOT-${label}-semantic-response.json"
  python3 - "$req" "$MODEL" <<'PY'
import json,sys
json.dump({
    "model": sys.argv[2],
    "messages": [{"role":"user","content":"Reply with exactly: TEXT_CONTROL_OK"}],
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
  python3 - "$resp" "$http" "$rc" "$label" <<'PY'
import json,sys
path,http,rc,label=sys.argv[1],sys.argv[2],int(sys.argv[3]),sys.argv[4]
ok=rc==0 and http=='200'
content=reasoning=finish=''
try:
    d=json.load(open(path)); c=(d.get('choices') or [{}])[0]; m=c.get('message') or {}
    content=str(m.get('content') or '').strip()
    reasoning=str(m.get('reasoning_content') or '').strip()
    finish=str(c.get('finish_reason') or '')
    combined=' '.join(x for x in (reasoning,content) if x)
    ok = ok and 'TEXT_CONTROL_OK' in combined
except Exception as e:
    print(f'eagle_final_{label}_semantic_parse_error={e!r}')
    ok=False
print(f'eagle_final_{label}_semantic_http={http}')
print(f'eagle_final_{label}_semantic_content={content!r}')
print(f'eagle_final_{label}_semantic_reasoning={reasoning!r}')
print(f'eagle_final_{label}_semantic_finish_reason={finish!r}')
print(f'eagle_final_{label}_semantic_pass={ok}')
raise SystemExit(0 if ok else 1)
PY
}

extract_elapsed() {
  python3 - "$1" <<'PY'
import re,sys
text=open(sys.argv[1], errors='replace').read()
m=re.findall(r'^stage_elapsed_seconds=([0-9.]+)$', text, re.M)
if not m:
    raise SystemExit(f'missing stage_elapsed_seconds in {sys.argv[1]}')
print(m[-1])
PY
}

make_candidate_gate
mkdir -p "$(dirname "$ROOT")"
LONG_LOG="${ROOT}-long.log"
EAGER_PERF_LOG="${ROOT}-perf-eager.log"
CANDIDATE_PERF_LOG="${ROOT}-perf-candidate.log"
rm -f "$LONG_LOG" "$EAGER_PERF_LOG" "$CANDIDATE_PERF_LOG"

echo '============================================================'
echo ' FINAL PRODUCTION GRAPH VALIDATION: PP3 + native MTP + multimodal'
echo ' target-verify graph: capture/replay OFF (eager)'
echo ' EAGLE draft-decode graph: ON'
echo ' EAGLE draft-extend graph: ON'
echo ' target/draft prefill CUDA graph: OFF'
echo " partition=${PARTITION} context=${CTX} pool=${MAX_TOTAL_TOKENS}"
echo '============================================================'

echo
echo '=== 1/4 EAGLE-GRAPH FULL 2x256K LONG STRESS ==='
STAGE=262000 DECODE_TOKENS=8 ROOT="${ROOT}-long-artifacts" \
  bash "$ROOT_DIR/qwen38_failfast_watch.sh" bash "$TMP_GATE" | tee "$LONG_LOG"
echo 'EAGLE_GRAPH_X2_256K_LONG_STRESS=PASS'

echo
echo '=== 2/4 POST-LONG TEXT + MULTIMODAL CHECK ==='
semantic_control long
PORT="$PORT" CONTAINER="$CONTAINER" ROOT="${ROOT}-mm" \
  bash "$ROOT_DIR/qwen38_gittensor_pp3_nvfp4_mm_smoke.sh"
echo 'EAGLE_GRAPH_TEXT_AND_MM_SEMANTICS=PASS'

echo
echo '=== 3/4 DECODE-HEAVY EAGER BASELINE ==='
STAGE=4096 DECODE_TOKENS=1024 ROOT="${ROOT}-perf-eager-artifacts" \
  bash "$ROOT_DIR/qwen38_failfast_watch.sh" \
  bash "$ROOT_DIR/qwen38_gittensor_pp3_nvfp4_256k_x2_gate.sh" | tee "$EAGER_PERF_LOG"
echo 'EAGLE_GRAPH_FINAL_EAGER_PERF_BASELINE=PASS'

echo
echo '=== 4/4 DECODE-HEAVY EAGLE-GRAPH CANDIDATE ==='
STAGE=4096 DECODE_TOKENS=1024 ROOT="${ROOT}-perf-candidate-artifacts" \
  bash "$ROOT_DIR/qwen38_failfast_watch.sh" bash "$TMP_GATE" | tee "$CANDIDATE_PERF_LOG"
semantic_control perf

echo
echo '=== FINAL PERFORMANCE COMPARISON ==='
EAGER_S="$(extract_elapsed "$EAGER_PERF_LOG")"
CANDIDATE_S="$(extract_elapsed "$CANDIDATE_PERF_LOG")"
python3 - "$EAGER_S" "$CANDIDATE_S" <<'PY'
import sys
eager=float(sys.argv[1]); candidate=float(sys.argv[2]); work=2048
speedup=(eager/candidate-1.0)*100.0
print(f'eagle_final_eager_elapsed_seconds={eager:.3f}')
print(f'eagle_final_candidate_elapsed_seconds={candidate:.3f}')
print(f'eagle_final_eager_effective_completion_tps={work/eager:.3f}')
print(f'eagle_final_candidate_effective_completion_tps={work/candidate:.3f}')
print(f'eagle_final_speedup_pct={speedup:.2f}')
print(f'eagle_final_perf_improved={speedup > 0.0}')
PY

echo 'QWEN38_EAGLE_GRAPH_FINAL_STRUCTURAL_PASS=True'
echo 'QWEN38_EAGLE_GRAPH_FINAL_SEMANTIC_PASS=True'
echo 'QWEN38_EAGLE_GRAPH_FINAL_MM_PASS=True'
echo 'QWEN38_GITTENSOR_PP3_NVFP4_X2_256K_EAGLE_GRAPH_FINAL=PASS'
echo "production_candidate_endpoint=http://127.0.0.1:${PORT}"
SUCCESS=1
