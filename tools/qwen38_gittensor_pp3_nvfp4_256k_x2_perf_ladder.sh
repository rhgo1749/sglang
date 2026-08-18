#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export IMAGE="${IMAGE:-sglang:qwen38-27b-pp-mtp-share}"
export MODEL="${MODEL:-gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090}"
export CONTAINER="${CONTAINER:-sglang-qwen38-gittensor-pp3}"
export PORT="${PORT:-30001}"

# Freeze the validated structural baseline.  This ladder changes only the two
# performance knobs we intentionally deferred: prefill chunk size, then decode
# CUDA Graph.  Capacity/partition/MTP/KV policy are not allowed to drift.
export CTX="${CTX:-262144}"
export CONTEXT_LENGTH="${CONTEXT_LENGTH:-262144}"
export PARALLEL="${PARALLEL:-2}"
export MAX_RUNNING="${MAX_RUNNING:-2}"
export MAX_MAMBA="${MAX_MAMBA:-2}"
export PARTITION="${PARTITION:-23,28,13}"
export MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.84}"
export MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-524288}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export ENABLE_MTP="${ENABLE_MTP:-1}"
export SPECULATIVE_ALGO="${SPECULATIVE_ALGO:-NEXTN}"
export SPECULATIVE_NUM_STEPS="${SPECULATIVE_NUM_STEPS:-3}"
export SPECULATIVE_EAGLE_TOPK="${SPECULATIVE_EAGLE_TOPK:-1}"
export SPECULATIVE_NUM_DRAFT_TOKENS="${SPECULATIVE_NUM_DRAFT_TOKENS:-4}"
export STAGE="${STAGE:-262000}"
export DECODE_TOKENS="${DECODE_TOKENS:-8}"
export REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-1200}"
export FAILFAST_STALL_SECONDS="${FAILFAST_STALL_SECONDS:-45}"
export FAILFAST_STALL_GPU_UTIL_MAX="${FAILFAST_STALL_GPU_UTIL_MAX:-5}"

CHUNK_STAGE="${CHUNK_STAGE:-1024}"
CG_JSON='{"decode":{"backend":"full","max_bs":2,"bs":[1,2]},"prefill":{"backend":"disabled"}}'

cleanup() {
  [[ -n "${TMP_GATE:-}" ]] && rm -f "$TMP_GATE" || true
}
trap cleanup EXIT

echo '============================================================'
echo ' PERF LADDER A: chunked prefill 1024 / CUDA Graph OFF'
echo '============================================================'
export CHUNKED_PREFILL_SIZE="$CHUNK_STAGE"
bash "$ROOT_DIR/qwen38_gittensor_pp3_nvfp4_256k_x2_probe.sh"
echo 'QWEN38_X2_256K_PERF_STAGE_A_CHUNK1024=PASS'

# The validated gate intentionally hard-codes --disable-cuda-graph.  Generate a
# temporary host-only variant for the experimental decode-only graph stage so
# the golden gate remains unchanged and can always be rerun verbatim.
TMP_GATE="$(mktemp /tmp/qwen38-x2-decode-cg.XXXXXX.sh)"
python3 - "$ROOT_DIR/qwen38_gittensor_pp3_nvfp4_256k_x2_gate.sh" "$TMP_GATE" "$CG_JSON" <<'PY'
from pathlib import Path
import sys
src, dst, cfg = map(str, sys.argv[1:])
text = Path(src).read_text()
needle = "    --disable-cuda-graph \\\n"
replacement = f"    --cuda-graph-config '{cfg}' \\\n"
if needle not in text:
    raise SystemExit("decode-CG perf ladder: --disable-cuda-graph anchor not found")
text = text.replace(needle, replacement, 1)
text = text.replace(
    " MTP ON / CUDA GRAPH OFF / SERVER WARMUP OFF",
    " MTP ON / DECODE CUDA GRAPH BS=1,2 / PREFILL GRAPH OFF / SERVER WARMUP OFF",
    1,
)
Path(dst).write_text(text)
PY
chmod +x "$TMP_GATE"

echo
echo '============================================================'
echo ' PERF LADDER B: chunked prefill 1024 / decode CUDA Graph BS=1,2'
echo '============================================================'
bash "$ROOT_DIR/qwen38_failfast_watch.sh" bash "$TMP_GATE"
echo 'QWEN38_X2_256K_PERF_STAGE_B_DECODE_CG=PASS'

echo
echo 'QWEN38_GITTENSOR_PP3_NVFP4_X2_256K_PERF_LADDER=PASS'
