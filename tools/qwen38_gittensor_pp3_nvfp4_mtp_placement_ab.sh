#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="${IMAGE:-sglang:qwen38-27b-pp-mtp-share}"
MODEL="${MODEL:-gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090}"
PORT="${PORT:-30001}"
PARTITION="${PARTITION:-23,28,13}"
PERF_STAGE="${PERF_STAGE:-4096}"
PERF_DECODE_TOKENS="${PERF_DECODE_TOKENS:-1024}"
ROOT="${ROOT:-/tmp/qwen38-mtp-placement-ab}"

# NOTE: Case C (MTP on a 5070 Ti) requires a real authoritative-sidecar/cutover
# path, not merely changing CUDA_VISIBLE_DEVICES. The current validated image
# keeps the authoritative native-MTP worker colocated with PP-last. This probe
# therefore runs the two currently-valid production comparisons and stops with
# an explicit marker instead of pretending that a 5070-sidecar result is valid.

mkdir -p "$ROOT"

extract_elapsed() {
  python3 - "$1" <<'PY'
import re,sys
m=re.findall(r'^stage_elapsed_seconds=([0-9.]+)$',open(sys.argv[1],errors='replace').read(),re.M)
if not m: raise SystemExit(f'missing stage_elapsed_seconds in {sys.argv[1]}')
print(m[-1])
PY
}

run_case() {
  local label="$1"; shift
  local log="$ROOT/${label}.log"
  echo "=== ${label} ==="
  STAGE="$PERF_STAGE" DECODE_TOKENS="$PERF_DECODE_TOKENS" ROOT="$ROOT/${label}-artifacts" \
    bash "$ROOT_DIR/qwen38_failfast_watch.sh" "$@" | tee "$log"
  extract_elapsed "$log"
}

echo '============================================================'
echo ' MTP PLACEMENT / ENABLEMENT A/B'
echo ' A: native MTP ON, authoritative draft colocated on PP-last (current 5060 Ti)'
echo ' B: MTP OFF, target-only PP3 baseline'
echo ' C: MTP ON with authoritative draft on a 5070 Ti -- requires dedicated cutover path'
echo " partition=${PARTITION} prompt=${PERF_STAGE} decode=${PERF_DECODE_TOKENS}"
echo '============================================================'

A_LOG="$ROOT/a_mtp_pp_last.log"
B_LOG="$ROOT/b_mtp_off.log"

# A: current validated graph-off native-MTP path.
STAGE="$PERF_STAGE" DECODE_TOKENS="$PERF_DECODE_TOKENS" ROOT="$ROOT/a-artifacts" \
  bash "$ROOT_DIR/qwen38_failfast_watch.sh" \
  bash "$ROOT_DIR/qwen38_gittensor_pp3_nvfp4_256k_x2_gate.sh" | tee "$A_LOG"
A_S="$(extract_elapsed "$A_LOG")"

# B: generate a target-only gate by replacing NEXTN with disabled speculative decoding.
TMP_B="$(mktemp /tmp/qwen38-mtp-off.XXXXXX.sh)"
trap 'rm -f "$TMP_B"' EXIT
python3 - "$ROOT_DIR/qwen38_gittensor_pp3_nvfp4_256k_x2_gate.sh" "$TMP_B" <<'PY'
from pathlib import Path
import sys
src,dst=sys.argv[1:]
s=Path(src).read_text()
s=s.replace('    --speculative-algorithm NEXTN \\\n','',1)
s=s.replace('    --speculative-num-steps 3 \\\n','',1)
s=s.replace('    --speculative-eagle-topk 1 \\\n','',1)
s=s.replace('    --speculative-num-draft-tokens 4 \\\n','',1)
s=s.replace(' MTP ON / CUDA GRAPH OFF / SERVER WARMUP OFF',' MTP OFF / TARGET-ONLY / CUDA GRAPH OFF',1)
Path(dst).write_text(s)
PY
chmod +x "$TMP_B"
STAGE="$PERF_STAGE" DECODE_TOKENS="$PERF_DECODE_TOKENS" ROOT="$ROOT/b-artifacts" \
  bash "$ROOT_DIR/qwen38_failfast_watch.sh" bash "$TMP_B" | tee "$B_LOG"
B_S="$(extract_elapsed "$B_LOG")"

python3 - "$A_S" "$B_S" <<'PY'
import sys
a,b=map(float,sys.argv[1:])
print(f'mtp_ab_current_mtp_pp_last_seconds={a:.3f}')
print(f'mtp_ab_mtp_off_seconds={b:.3f}')
print(f'mtp_ab_mtp_on_vs_off_speedup_pct={(b/a-1.0)*100.0:.2f}')
print(f'mtp_ab_mtp_on_faster={a < b}')
PY

echo 'MTP_5070_PLACEMENT_STATUS=NOT_YET_VALIDATED_REQUIRES_AUTHORITATIVE_SIDECAR_CUTOVER'
echo 'QWEN38_MTP_PLACEMENT_AB_PARTIAL=PASS'

echo '=== RESTORE GRAPH-OFF NATIVE-MTP PRODUCTION BASELINE ==='
IMAGE="$IMAGE" MODEL="$MODEL" PORT="$PORT" \
  bash "$ROOT_DIR/qwen38_gittensor_pp3_nvfp4_256k_x2_serve.sh"
