#!/usr/bin/env bash
set -euo pipefail

# Long-context runtime-headroom probe.
#
# Observed baselines:
#   0.990 / chunk 2048 -> raw 3x256K capacity PASS, short functional PASS,
#                         but real 3x64K prefill OOM.
#   0.985 / chunk 1024 -> capacity 793856 PASS, short functional PASS,
#                         but PP2 reached ~4 MiB free and failed a 22 MiB alloc.
#   0.983 / chunk  512 -> capacity 790272 PASS, short functional PASS,
#                         but PP2 reached 24.06 MiB free and failed a 30 MiB alloc.
#
# Capacity fell by 3584 tokens when mem_fraction_static moved 0.985 -> 0.983.
# Extrapolating one more 0.002 step predicts ~786688 tokens at 0.981, only
# ~256 tokens above the mandatory 3*262144 requirement. This is therefore the
# final practical static-KV boundary probe. Halve the prefill chunk once more
# to reduce the transient working set. The capacity gate remains mandatory.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PARTITION="${PARTITION:-19,23,22}"
export MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.981}"
export CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-256}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo '============================================================'
echo ' QWEN3.8 LONG-CONTEXT RUNTIME HEADROOM PROBE v3'
echo " partition=${PARTITION}"
echo " mem_fraction_static=${MEM_FRACTION_STATIC}"
echo " chunked_prefill_size=${CHUNKED_PREFILL_SIZE}"
echo " pytorch_cuda_alloc_conf=${PYTORCH_CUDA_ALLOC_CONF}"
echo ' target: preserve >=786432 KV tokens and survive real 3x64K+'
echo ' note: 0.981 is the predicted knife-edge capacity boundary'
echo '============================================================'

echo '=== PHASE 1: capacity + short functional gate ==='
bash "$ROOT_DIR/qwen38_vanilla_pp3_nvfp4_final_gate.sh"

echo
echo '=== PHASE 2: real parallel-3 long-context stress gate ==='
bash "$ROOT_DIR/qwen38_vanilla_pp3_nvfp4_long_stress_gate.sh"

echo
echo 'QWEN38_PP3_NVFP4_LONG_HEADROOM_PROBE=PASS'
