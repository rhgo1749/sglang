#!/usr/bin/env bash
set -euo pipefail

# Corrected retry wrapper for qwen38_mtp_fp8_parallel3_validate.sh.
#
# The first Mamba8 / mem_fraction_static=0.80 run measured a 57,152-token
# target FP8 pool, then exited before server readiness because the validator
# accidentally reused its final single-request 64K requirement as the
# authoritative cutover *boot* gate.  The cutover is intentionally
# parallel-aware and normally boots max_running>1 with a 32,768-token minimum;
# the validator itself performs the stricter exact 3-way physical-capacity gate
# after startup.
#
# Keep Mamba8: reducing it further would trade away request-level recurrent
# state headroom.  Instead raise the target static allocation budget from 0.80
# to 0.83.  On this 16 GiB target rank that adds roughly 0.46 GiB of static
# budget.  The measured FP8 KV pool uses about 16 KiB/token, so this is enough
# headroom to cross the validator's exact 75,840-slot 3 x (25,194 + 64) gate
# while still leaving substantially more than 2 GiB free before CUDA-graph
# capture in the observed 0.80 run.

REPO="${HOME}/projects/sglang-fork"
BASE="${REPO}/tools/qwen38_mtp_fp8_parallel3_validate.sh"

if [[ ! -f "$BASE" ]]; then
  echo "ERROR: base validator not found: $BASE" >&2
  exit 1
fi

# 32K is only the startup/cutover admission floor.  The base validator still
# enforces P3_REQUIRED (75,840 slots with its defaults) before it sends the
# parallel stress, and then executes the real 65,520+8 single-request test.
export MTP_FP8_MIN_POOL="${MTP_FP8_MIN_POOL:-32768}"
export MTP_FP8_MEM_FRACTION_STATIC="${MTP_FP8_MEM_FRACTION_STATIC:-0.83}"
export MTP_FP8_MAX_MAMBA="${MTP_FP8_MAX_MAMBA:-8}"

echo '=== CORRECTED FP8 PARALLEL-3 RETRY ==='
echo "mem_fraction_static=${MTP_FP8_MEM_FRACTION_STATIC}"
echo "max_mamba_cache_size=${MTP_FP8_MAX_MAMBA}"
echo "cutover_boot_min_pool=${MTP_FP8_MIN_POOL}"
echo 'post-start capacity gate remains exact P3_REQUIRED in the base validator'

exec bash "$BASE"
