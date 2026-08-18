#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="${IMAGE:-sglang:qwen38-27b-pp-mtp-share}"
MODEL="${MODEL:-gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090}"
PORT="${PORT:-30001}"
PERF_STAGE="${PERF_STAGE:-4096}"
PERF_DECODE_TOKENS="${PERF_DECODE_TOKENS:-1024}"
ROOT="${ROOT:-/tmp/qwen38-mtp5070-partition-sweep-v2}"

# Mapping for every candidate:
#   physical GPU0 RTX 5070 Ti -> PP0
#   physical GPU1 RTX 5060 Ti -> PP1
#   physical GPU2 RTX 5070 Ti -> PP2 + colocated native MTP
#
# Observed capacity boundary:
#   26,22,16 => PP0 profiles only 429376 target tokens, below required 524288.
# Therefore keep PP0 at the validated 23-layer level and shift target work from
# the slower middle 5060 Ti stage toward the final 5070 Ti + MTP stage.
#
# Baselines:
#   A: 23,28,13 with PP-last/MTP on 5060 Ti = 98.922 s
#   C: 23,28,13 with PP-last/MTP on 5070 Ti = 107.629 s
#
# Stop condition: if none of these bounded candidates beats A, close the
# MTP@5070 placement family rather than continuing a broad partition search.

PARTITIONS=(
  "23,22,19"
  "23,21,20"
  "23,20,21"
)

mkdir -p "$ROOT"
RESULTS="$ROOT/results.tsv"
: > "$RESULTS"

restore_baseline() {
  echo '=== RESTORE GRAPH-OFF NATIVE-MTP PRODUCTION BASELINE ==='
  IMAGE="$IMAGE" MODEL="$MODEL" PORT="$PORT" \
    bash "$ROOT_DIR/qwen38_gittensor_pp3_nvfp4_256k_x2_serve.sh" || true
}
trap restore_baseline EXIT

echo '============================================================'
echo ' MTP@5070 PARTITION SWEEP V2'
echo ' mapping: 5070(PP0) / 5060(PP1) / 5070(PP2+MTP)'
echo ' strategy: hold PP0=23, reduce PP1, grow PP2'
echo " prompt=${PERF_STAGE} decode=${PERF_DECODE_TOKENS}"
echo ' reference_A_5060_last_seconds=98.922'
echo ' reference_C_23_28_13_seconds=107.629'
echo ' known_capacity_fail_26_22_16_pp0_tokens=429376'
echo '============================================================'

for p in "${PARTITIONS[@]}"; do
  label="${p//,/_}"
  log="$ROOT/${label}.log"
  echo
  echo "=== CANDIDATE partition=${p} ==="

  set +e
  IMAGE="$IMAGE" MODEL="$MODEL" PORT="$PORT" \
  PARTITION="$p" PERF_STAGE="$PERF_STAGE" PERF_DECODE_TOKENS="$PERF_DECODE_TOKENS" \
  RUN_ONLY_C=1 ROOT="$ROOT/run-${label}" \
    bash "$ROOT_DIR/qwen38_gittensor_pp3_nvfp4_mtp_placement_ab.sh" \
    2>&1 | tee "$log"
  rc=${PIPESTATUS[0]}
  set -e

  if (( rc != 0 )); then
    printf '%s\tFAIL\t-\n' "$p" >> "$RESULTS"
    echo "mtp5070_sweep_${label}_status=FAIL"
    continue
  fi

  elapsed="$(python3 - "$log" <<'PY'
import re,sys
text=open(sys.argv[1],errors='replace').read()
m=re.findall(r'^mtp_abc_mtp_5070_pp_last_seconds=([0-9.]+)$',text,re.M)
if not m:
    raise SystemExit(2)
print(m[-1])
PY
)"
  printf '%s\tPASS\t%s\n' "$p" "$elapsed" >> "$RESULTS"
  echo "mtp5070_sweep_${label}_status=PASS"
  echo "mtp5070_sweep_${label}_seconds=${elapsed}"
done

echo
echo '=== SWEEP SUMMARY ==='
python3 - "$RESULTS" <<'PY'
import sys
rows=[]
for line in open(sys.argv[1]):
    p,status,val=line.rstrip('\n').split('\t')
    if status=='PASS':
        rows.append((float(val),p))
    print(f'mtp5070_partition_{p.replace(",","_")}_status={status}')
    if status=='PASS':
        print(f'mtp5070_partition_{p.replace(",","_")}_seconds={float(val):.3f}')
if not rows:
    print('mtp5070_partition_best=NONE')
    print('mtp5070_partition_family_beats_A=False')
    raise SystemExit(1)
rows.sort()
best_s,best_p=rows[0]
a=98.922
c=107.629
print(f'mtp5070_partition_best={best_p}')
print(f'mtp5070_partition_best_seconds={best_s:.3f}')
print(f'mtp5070_partition_best_vs_A_speedup_pct={(a/best_s-1.0)*100.0:.2f}')
print(f'mtp5070_partition_best_vs_C_speedup_pct={(c/best_s-1.0)*100.0:.2f}')
print(f'mtp5070_partition_family_beats_A={best_s < a}')
PY

echo 'QWEN38_MTP5070_PARTITION_SWEEP=PASS'
