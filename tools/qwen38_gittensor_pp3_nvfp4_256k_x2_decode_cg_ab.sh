#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export IMAGE="${IMAGE:-sglang:qwen38-27b-pp-mtp-share}"
export MODEL="${MODEL:-gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090}"
export CONTAINER="${CONTAINER:-sglang-qwen38-gittensor-pp3}"
export PORT="${PORT:-30001}"

# Freeze the validated structural/perf baseline.  This A/B changes only decode
# CUDA Graph.  Keep full 2x256K capacity while using a modest prompt and a long
# forced decode so graph replay benefit is visible instead of being buried by a
# 262K prefill.
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
export ENABLE_MTP="${ENABLE_MTP:-1}"
export SPECULATIVE_ALGO="${SPECULATIVE_ALGO:-NEXTN}"
export SPECULATIVE_NUM_STEPS="${SPECULATIVE_NUM_STEPS:-3}"
export SPECULATIVE_EAGLE_TOPK="${SPECULATIVE_EAGLE_TOPK:-1}"
export SPECULATIVE_NUM_DRAFT_TOKENS="${SPECULATIVE_NUM_DRAFT_TOKENS:-4}"

export STAGE="${AB_PROMPT_TOKENS:-4096}"
export DECODE_TOKENS="${AB_DECODE_TOKENS:-1024}"
export REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-1200}"
export FAILFAST_STALL_SECONDS="${FAILFAST_STALL_SECONDS:-45}"
export FAILFAST_STALL_GPU_UTIL_MAX="${FAILFAST_STALL_GPU_UTIL_MAX:-5}"

CG_JSON='{"decode":{"backend":"full","max_bs":2,"bs":[1,2]},"prefill":{"backend":"disabled"}}'
ROOT="${AB_ROOT:-/tmp/qwen38-x2-decode-cg-ab}"
OFF_LOG="${ROOT}-off.log"
ON_LOG="${ROOT}-on.log"
TMP_GATE=""

cleanup() {
  [[ -n "${TMP_GATE:-}" ]] && rm -f "$TMP_GATE" || true
}
trap cleanup EXIT

make_cg_gate() {
  TMP_GATE="$(mktemp /tmp/qwen38-x2-decode-cg-ab.XXXXXX.sh)"
  python3 - "$ROOT_DIR/qwen38_gittensor_pp3_nvfp4_256k_x2_gate.sh" "$TMP_GATE" "$CG_JSON" <<'PY'
from pathlib import Path
import sys
src, dst, cfg = map(str, sys.argv[1:])
text = Path(src).read_text()
needle = "    --disable-cuda-graph \\\n"
replacement = f"    --cuda-graph-config '{cfg}' \\\n"
if needle not in text:
    raise SystemExit("decode-CG A/B: --disable-cuda-graph anchor not found")
text = text.replace(needle, replacement, 1)
text = text.replace(
    " MTP ON / CUDA GRAPH OFF / SERVER WARMUP OFF",
    " MTP ON / DECODE CUDA GRAPH BS=1,2 / PREFILL GRAPH OFF / SERVER WARMUP OFF",
    1,
)
Path(dst).write_text(text)
PY
  chmod +x "$TMP_GATE"
}

extract_elapsed() {
  python3 - "$1" <<'PY'
import re,sys
text=open(sys.argv[1], errors='replace').read()
m=re.findall(r'^stage_elapsed_seconds=([0-9.]+)$', text, re.M)
if not m:
    raise SystemExit(f"missing stage_elapsed_seconds in {sys.argv[1]}")
print(m[-1])
PY
}

echo '============================================================'
echo ' DECODE CUDA GRAPH A/B'
echo " prompt_tokens_per_request=${STAGE}"
echo " forced_decode_tokens_per_request=${DECODE_TOKENS}"
echo ' parallel=2 / chunk=1024 / full 524288 token pool'
echo '============================================================'

echo
echo '=== A: CUDA GRAPH OFF ==='
rm -f "$OFF_LOG" "$ON_LOG"
bash "$ROOT_DIR/qwen38_failfast_watch.sh" \
  bash "$ROOT_DIR/qwen38_gittensor_pp3_nvfp4_256k_x2_gate.sh" | tee "$OFF_LOG"
echo 'QWEN38_DECODE_CG_AB_OFF=PASS'

make_cg_gate

echo
echo '=== B: DECODE CUDA GRAPH BS=1,2 / PREFILL GRAPH OFF ==='
bash "$ROOT_DIR/qwen38_failfast_watch.sh" bash "$TMP_GATE" | tee "$ON_LOG"
echo 'QWEN38_DECODE_CG_AB_ON=PASS'

OFF_S="$(extract_elapsed "$OFF_LOG")"
ON_S="$(extract_elapsed "$ON_LOG")"
python3 - "$OFF_S" "$ON_S" "$DECODE_TOKENS" <<'PY'
import sys
off=float(sys.argv[1]); on=float(sys.argv[2]); dec=int(sys.argv[3])
work=2*dec
speedup=(off/on-1.0)*100.0
print(f'decode_ab_off_elapsed_seconds={off:.3f}')
print(f'decode_ab_on_elapsed_seconds={on:.3f}')
print(f'decode_ab_off_effective_completion_tps={work/off:.3f}')
print(f'decode_ab_on_effective_completion_tps={work/on:.3f}')
print(f'decode_ab_cuda_graph_speedup_pct={speedup:.2f}')
PY

echo 'QWEN38_GITTENSOR_PP3_NVFP4_X2_256K_DECODE_CG_AB=PASS'
