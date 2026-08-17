#!/usr/bin/env bash
set -euo pipefail

# A/B probe for gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090.
# Reuses the validated vanilla PP3 gate/stress harness while keeping the
# RadixArk baseline untouched.
#
# 192000 is used for the final stress stage instead of 196600 so the request
# retains several thousand tokens of decode headroom below context_len=196608.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export MODEL="${MODEL:-gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090}"
export CONTAINER="${CONTAINER:-sglang-qwen38-gittensor-pp3}"
export CTX="${CTX:-196608}"
export CONTEXT_LENGTH="${CONTEXT_LENGTH:-196608}"
export STAGES="${STAGES:-65536 131072 192000}"
export PARTITION="${PARTITION:-19,23,22}"
export MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.89}"
export CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-1024}"
export MAX_RUNNING="${MAX_RUNNING:-3}"
export MAX_MAMBA="${MAX_MAMBA:-3}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo '============================================================'
echo ' QWEN3.8 GITTENSOR RTX5090 NVFP4 PP3 3x192K A/B PROBE'
echo " model=${MODEL}"
echo " context_length=${CTX}"
echo " required_kv_tokens=$((CTX * MAX_RUNNING))"
echo " partition=${PARTITION}"
echo " mem_fraction_static=${MEM_FRACTION_STATIC}"
echo " chunked_prefill_size=${CHUNKED_PREFILL_SIZE}"
echo " stress_stages=${STAGES}"
echo ' MTP OFF / CUDA GRAPH OFF'
echo '============================================================'

bash "$ROOT_DIR/qwen38_vanilla_pp3_nvfp4_final_gate.sh"

echo
echo '=== LOAD / MEMORY SUMMARY ==='
docker logs "$CONTAINER" 2>&1 | \
  grep -Ei 'Load weight|weight.*memory|model.*memory|KV Cache|Mamba Cache|Memory pool end|max_total_num_tokens|available_gpu_mem' | \
  tail -160 || true

echo
echo '=== REAL PARALLEL-3 LONG-CONTEXT STRESS ==='
bash "$ROOT_DIR/qwen38_vanilla_pp3_nvfp4_long_stress_gate.sh"

echo
echo 'QWEN38_GITTENSOR_PP3_NVFP4_3X192K_PROBE=PASS'
