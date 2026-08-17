#!/usr/bin/env bash
set -euo pipefail

# Practical 3x192K runtime profile.
#
# 3x256K was proven to be a knife-edge static-capacity configuration.
# Moving to 192K reduced the required shared KV pool to 589824 tokens.
#
# At mem_fraction_static=0.92 the target pool was 679360 tokens and both
# real 3x64K and 3x128K completed, but the 3x196600 stage OOMed during
# prefill while PP0 missed a 158 MiB allocation by only ~10 MiB.
# That profile therefore still over-reserved KV by 89536 tokens.
#
# Use 0.89 to trade part of that surplus KV pool for several hundred MiB
# of runtime/pre-fill headroom while retaining a practical 1024-token chunk.
# The capacity gate remains mandatory before the long stress test starts.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export CTX="${CTX:-196608}"
export CONTEXT_LENGTH="${CONTEXT_LENGTH:-196608}"
export STAGES="${STAGES:-65536 131072 196600}"
export PARTITION="${PARTITION:-19,23,22}"
export MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.89}"
export CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-1024}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo '============================================================'
echo ' QWEN3.8 PRACTICAL PP3 + NVFP4 3x192K PROFILE v2'
echo " context_length=${CTX}"
echo " required_kv_tokens=$((CTX * 3))"
echo " partition=${PARTITION}"
echo " mem_fraction_static=${MEM_FRACTION_STATIC}"
echo " chunked_prefill_size=${CHUNKED_PREFILL_SIZE}"
echo " pytorch_cuda_alloc_conf=${PYTORCH_CUDA_ALLOC_CONF}"
echo " stress_stages=${STAGES}"
echo ' target: preserve 3x192K capacity with meaningful runtime/MTP headroom'
echo '============================================================'

echo '=== PHASE 1: 3x192K capacity + short functional gate ==='
bash "$ROOT_DIR/qwen38_vanilla_pp3_nvfp4_final_gate.sh"

echo
echo '=== PHASE 2: real parallel-3 long-context stress gate ==='
bash "$ROOT_DIR/qwen38_vanilla_pp3_nvfp4_long_stress_gate.sh"

echo
echo 'QWEN38_PP3_NVFP4_3X192K_PROFILE=PASS'
