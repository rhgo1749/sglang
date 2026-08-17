#!/usr/bin/env bash
set -euo pipefail

# Practical 3x192K runtime profile.
#
# 3x256K was proven to be a knife-edge static-capacity configuration:
#   - 0.981 / chunk 256 still exposed only 192 spare KV tokens
#   - real 3x64K prefill still OOMed on PP2 while requesting runtime scratch
#
# Stop tuning the 256K boundary. 192K cuts the required shared KV pool to
# 589824 tokens and intentionally gives runtime prefill substantially more
# headroom. Restore a practical 1024-token prefill chunk for better throughput.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export CTX="${CTX:-196608}"
export CONTEXT_LENGTH="${CONTEXT_LENGTH:-196608}"
export STAGES="${STAGES:-65536 131072 196600}"
export PARTITION="${PARTITION:-19,23,22}"
export MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.92}"
export CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-1024}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo '============================================================'
echo ' QWEN3.8 PRACTICAL PP3 + NVFP4 3x192K PROFILE'
echo " context_length=${CTX}"
echo " required_kv_tokens=$((CTX * 3))"
echo " partition=${PARTITION}"
echo " mem_fraction_static=${MEM_FRACTION_STATIC}"
echo " chunked_prefill_size=${CHUNKED_PREFILL_SIZE}"
echo " pytorch_cuda_alloc_conf=${PYTORCH_CUDA_ALLOC_CONF}"
echo " stress_stages=${STAGES}"
echo ' target: reliable runtime headroom, not knife-edge 256K capacity'
echo '============================================================'

echo '=== PHASE 1: 3x192K capacity + short functional gate ==='
bash "$ROOT_DIR/qwen38_vanilla_pp3_nvfp4_final_gate.sh"

echo
echo '=== PHASE 2: real parallel-3 long-context stress gate ==='
bash "$ROOT_DIR/qwen38_vanilla_pp3_nvfp4_long_stress_gate.sh"

echo
echo 'QWEN38_PP3_NVFP4_3X192K_PROFILE=PASS'
