#!/usr/bin/env bash
set -euo pipefail

# Conservative long-context probe profile.
# Keeps the validated PP split, but gives runtime prefill scratch more room by:
#   1) reserving slightly less static KV memory
#   2) halving the chunked prefill working set
# The final capacity gate still has to prove >= 3 * 262144 tokens before
# the real long-context stress test is attempted.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PARTITION="${PARTITION:-19,23,22}"
export MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.985}"
export CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-1024}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo '============================================================'
echo ' QWEN3.8 LONG-CONTEXT RUNTIME HEADROOM PROBE'
echo " partition=${PARTITION}"
echo " mem_fraction_static=${MEM_FRACTION_STATIC}"
echo " chunked_prefill_size=${CHUNKED_PREFILL_SIZE}"
echo " pytorch_cuda_alloc_conf=${PYTORCH_CUDA_ALLOC_CONF}"
echo '============================================================'

echo '=== PHASE 1: capacity + short functional gate ==='
bash "$ROOT_DIR/qwen38_vanilla_pp3_nvfp4_final_gate.sh"

echo
echo '=== PHASE 2: real parallel-3 long-context stress gate ==='
bash "$ROOT_DIR/qwen38_vanilla_pp3_nvfp4_long_stress_gate.sh"

echo
echo 'QWEN38_PP3_NVFP4_LONG_HEADROOM_PROBE=PASS'
