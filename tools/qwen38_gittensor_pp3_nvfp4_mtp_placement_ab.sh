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
RUN_ONLY_C="${RUN_ONLY_C:-0}"

# Physical GPU layout on this host:
#   GPU0 = RTX 5070 Ti
#   GPU1 = RTX 5060 Ti
#   GPU2 = RTX 5070 Ti
#
# Validated production mapping is device=0,2,1, so logical CUDA2 / PP-last is
# physical GPU1 (5060 Ti) and owns the colocated native-MTP worker.
#
# Case C deliberately changes only the Docker GPU order to device=0,1,2.
# Therefore logical CUDA2 / PP-last (and the ordinary colocated MTP worker)
# becomes physical GPU2 (5070 Ti), while logical CUDA1 / PP1 becomes the 5060 Ti.
# This is a real supported PP3 execution path and avoids pretending that the old
# PP=1 authoritative-sidecar prototype is valid for PP3.  It is NOT a pure
# draft-only relocation: the PP-last target stage moves with MTP.  Keep the same
# partition first; if C wins, tune the partition with the 5060 now on PP1.

mkdir -p "$ROOT"
TMP_B=""
TMP_C=""

cleanup() {
  [[ -n "${TMP_B:-}" ]] && rm -f "$TMP_B" || true
  [[ -n "${TMP_C:-}" ]] && rm -f "$TMP_C" || true
}
trap cleanup EXIT

extract_elapsed() {
  python3 - "$1" <<'PY'
import re,sys
m=re.findall(r'^stage_elapsed_seconds=([0-9.]+)$',open(sys.argv[1],errors='replace').read(),re.M)
if not m: raise SystemExit(f'missing stage_elapsed_seconds in {sys.argv[1]}')
print(m[-1])
PY
}

make_mtp_off_gate() {
  TMP_B="$(mktemp /tmp/qwen38-mtp-off.XXXXXX.sh)"
  python3 - "$ROOT_DIR/qwen38_gittensor_pp3_nvfp4_256k_x2_gate.sh" "$TMP_B" <<'PY'
from pathlib import Path
import sys
src,dst=sys.argv[1:]
s=Path(src).read_text()
s=s.replace('    --speculative-algorithm "$SPECULATIVE_ALGO" \\\n','',1)
s=s.replace('    --speculative-num-steps "$SPECULATIVE_NUM_STEPS" \\\n','',1)
s=s.replace('    --speculative-eagle-topk "$SPECULATIVE_EAGLE_TOPK" \\\n','',1)
s=s.replace('    --speculative-num-draft-tokens "$SPECULATIVE_NUM_DRAFT_TOKENS" \\\n','',1)
# The base gate constructs MTP_ARGS dynamically. Disabling ENABLE_MTP is the
# authoritative switch; keep the textual removals above harmless across older
# exact-image gate revisions.
s='ENABLE_MTP="${ENABLE_MTP:-0}"'.join(s.split('ENABLE_MTP="${ENABLE_MTP:-1}"',1)) if 'ENABLE_MTP="${ENABLE_MTP:-1}"' in s else s
s=s.replace(' MTP ON / CUDA GRAPH OFF / SERVER WARMUP OFF',' MTP OFF / TARGET-ONLY / CUDA GRAPH OFF',1)
Path(dst).write_text(s)
PY
  chmod +x "$TMP_B"
}

make_5070_last_gate() {
  TMP_C="$(mktemp /tmp/qwen38-mtp-5070-last.XXXXXX.sh)"
  python3 - "$ROOT_DIR/qwen38_gittensor_pp3_nvfp4_256k_x2_gate.sh" "$TMP_C" <<'PY'
from pathlib import Path
import sys
src,dst=sys.argv[1:]
s=Path(src).read_text()
old="  --gpus '\"device=0,2,1\"' \\\n"
new="  --gpus '\"device=0,1,2\"' \\\n"
if s.count(old) != 1:
    raise SystemExit(f'5070-last gate: expected exactly one GPU-order anchor, got {s.count(old)}')
s=s.replace(old,new,1)
s=s.replace(
    ' MTP ON / CUDA GRAPH OFF / SERVER WARMUP OFF',
    ' MTP ON / PP-LAST+MTP ON PHYSICAL GPU2 RTX5070TI / CUDA GRAPH OFF',
    1,
)
Path(dst).write_text(s)
PY
  chmod +x "$TMP_C"
}

run_gate() {
  local label="$1" gate="$2" log="$3"
  echo
  echo "=== ${label} ==="
  STAGE="$PERF_STAGE" DECODE_TOKENS="$PERF_DECODE_TOKENS" \
  PARTITION="$PARTITION" ROOT="$ROOT/${label}-artifacts" \
    bash "$ROOT_DIR/qwen38_failfast_watch.sh" bash "$gate" | tee "$log"
}

echo '============================================================'
echo ' MTP PLACEMENT / ENABLEMENT A/B/C'
echo ' A: PP-last + native MTP on physical GPU1 / RTX 5060 Ti (device=0,2,1)'
echo ' B: MTP OFF / target-only PP3'
echo ' C: PP-last + native MTP on physical GPU2 / RTX 5070 Ti (device=0,1,2)'
echo '    NOTE: C also moves PP1 target stage onto the 5060 Ti; same partition first.'
echo " partition=${PARTITION} prompt=${PERF_STAGE} decode=${PERF_DECODE_TOKENS}"
echo " run_only_c=${RUN_ONLY_C}"
echo '============================================================'

A_LOG="$ROOT/a_mtp_5060_last.log"
B_LOG="$ROOT/b_mtp_off.log"
C_LOG="$ROOT/c_mtp_5070_last.log"
A_S=""
B_S=""

if [[ "$RUN_ONLY_C" != "1" ]]; then
  STAGE="$PERF_STAGE" DECODE_TOKENS="$PERF_DECODE_TOKENS" \
  PARTITION="$PARTITION" ROOT="$ROOT/a-artifacts" \
    bash "$ROOT_DIR/qwen38_failfast_watch.sh" \
    bash "$ROOT_DIR/qwen38_gittensor_pp3_nvfp4_256k_x2_gate.sh" | tee "$A_LOG"
  A_S="$(extract_elapsed "$A_LOG")"

  make_mtp_off_gate
  STAGE="$PERF_STAGE" DECODE_TOKENS="$PERF_DECODE_TOKENS" \
  PARTITION="$PARTITION" ROOT="$ROOT/b-artifacts" \
    bash "$ROOT_DIR/qwen38_failfast_watch.sh" bash "$TMP_B" | tee "$B_LOG"
  B_S="$(extract_elapsed "$B_LOG")"
fi

make_5070_last_gate
run_gate c_mtp_5070_last "$TMP_C" "$C_LOG"
C_S="$(extract_elapsed "$C_LOG")"

if [[ "$RUN_ONLY_C" == "1" ]]; then
  python3 - "$C_S" <<'PY'
import sys
c=float(sys.argv[1])
print(f'mtp_abc_mtp_5070_pp_last_seconds={c:.3f}')
print('mtp_abc_case_c_mapping=physical0_5070_PP0__physical1_5060_PP1__physical2_5070_PP2_MTP')
print('mtp_abc_case_c_is_pure_draft_relocation=False')
PY
else
  python3 - "$A_S" "$B_S" "$C_S" <<'PY'
import sys
a,b,c=map(float,sys.argv[1:])
print(f'mtp_abc_mtp_5060_pp_last_seconds={a:.3f}')
print(f'mtp_abc_mtp_off_seconds={b:.3f}')
print(f'mtp_abc_mtp_5070_pp_last_seconds={c:.3f}')
print(f'mtp_abc_mtp_5060_vs_off_speedup_pct={(b/a-1.0)*100.0:.2f}')
print(f'mtp_abc_5070_last_vs_5060_last_speedup_pct={(a/c-1.0)*100.0:.2f}')
print(f'mtp_abc_5070_last_vs_off_speedup_pct={(b/c-1.0)*100.0:.2f}')
print(f'mtp_abc_best={min((a,"MTP_5060_LAST"),(b,"MTP_OFF"),(c,"MTP_5070_LAST"))[1]}')
print('mtp_abc_case_c_mapping=physical0_5070_PP0__physical1_5060_PP1__physical2_5070_PP2_MTP')
print('mtp_abc_case_c_is_pure_draft_relocation=False')
PY
fi

echo 'QWEN38_MTP_PLACEMENT_ABC=PASS'

echo '=== RESTORE GRAPH-OFF NATIVE-MTP PRODUCTION BASELINE ==='
IMAGE="$IMAGE" MODEL="$MODEL" PORT="$PORT" \
  bash "$ROOT_DIR/qwen38_gittensor_pp3_nvfp4_256k_x2_serve.sh"
