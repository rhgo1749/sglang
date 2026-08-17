#!/usr/bin/env bash
set -euo pipefail

# Long-context runtime-headroom probe.
#
# Observed baselines:
#   0.990 / chunk 2048 -> raw 3x256K capacity PASS, short functional PASS,
#                         but real 3x64K prefill OOM.
#   0.985 / chunk 1024 -> capacity 793856 PASS, short functional PASS,
#                         but PP2 reached ~4 MiB free and failed a 22 MiB alloc.
#
# This profile moves only a small additional slice from static KV reservation
# to runtime scratch while halving the prefill chunk again.  The capacity gate
# remains mandatory: if >= 3*262144 is not available, phase 2 will not run.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PARTITION="${PARTITION:-19,23,22}"
export MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.983}"
export CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-512}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo '============================================================'
echo ' QWEN3.8 LONG-CONTEXT RUNTIME HEADROOM PROBE v2'
echo " partition=${PARTITION}"
echo " mem_fraction_static=${MEM_FRACTION_STATIC}"
echo " chunked_prefill_size=${CHUNKED_PREFILL_SIZE}"
echo " pytorch_cuda_alloc_conf=${PYTORCH_CUDA_ALLOC_CONF}"
echo ' target: preserve >=786432 KV tokens and survive real 3x64K+'
echo '============================================================'

echo '=== PHASE 1: capacity + short functional gate ==='
bash "$ROOT_DIR/qwen38_vanilla_pp3_nvfp4_final_gate.sh"

echo
echo '=== PHASE 2: real parallel-3 long-context stress gate ==='
bash "$ROOT_DIR/qwen38_vanilla_pp3_nvfp4_long_stress_gate.sh"

echo
echo 'QWEN38_PP3_NVFP4_LONG_HEADROOM_PROBE=PASS'
