#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export MODEL="${MODEL:-gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090}"
export CONTAINER="${CONTAINER:-sglang-qwen38-gittensor-pp3}"

export CTX="${CTX:-196608}"
export CONTEXT_LENGTH="${CONTEXT_LENGTH:-196608}"

# Vanilla validated baseline.
export PARTITION="${PARTITION:-19,24,21}"
export MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.84}"
export MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-593920}"

export CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-1024}"
export MAX_RUNNING="${MAX_RUNNING:-3}"
export MAX_MAMBA="${MAX_MAMBA:-3}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Long stress defaults to the only stage we still care about.
export STAGES="${STAGES:-192000}"
export RUN_LONG_STRESS="${RUN_LONG_STRESS:-1}"

# Native checkpoint MTP.
export ENABLE_MTP="${ENABLE_MTP:-0}"
export SPECULATIVE_ALGO="${SPECULATIVE_ALGO:-NEXTN}"
export SPECULATIVE_NUM_STEPS="${SPECULATIVE_NUM_STEPS:-3}"
export SPECULATIVE_EAGLE_TOPK="${SPECULATIVE_EAGLE_TOPK:-1}"
export SPECULATIVE_NUM_DRAFT_TOKENS="${SPECULATIVE_NUM_DRAFT_TOKENS:-4}"

# auto:
#   vanilla -> capacity must satisfy CTX * parallel
#   MTP     -> allow smoke test even if speculative pools reduce capacity
export CAPACITY_GATE="${CAPACITY_GATE:-auto}"

if [[ "$ENABLE_MTP" == "1" ]]; then
    MTP_DESC="ON algo=${SPECULATIVE_ALGO} steps=${SPECULATIVE_NUM_STEPS} topk=${SPECULATIVE_EAGLE_TOPK} draft=${SPECULATIVE_NUM_DRAFT_TOKENS}"
else
    MTP_DESC="OFF"
fi

echo '============================================================'
echo ' QWEN3.8 GITTENSOR NVFP4 PP3 PROBE'
echo " model=${MODEL}"
echo " context_length=${CTX}"
echo " required_target_tokens=$((CTX * MAX_RUNNING))"
echo " partition=${PARTITION}"
echo " mem_fraction_static=${MEM_FRACTION_STATIC}"
echo " max_total_tokens=${MAX_TOTAL_TOKENS}"
echo " chunked_prefill_size=${CHUNKED_PREFILL_SIZE}"
echo " stress_stages=${STAGES}"
echo " run_long_stress=${RUN_LONG_STRESS}"
echo " capacity_gate=${CAPACITY_GATE}"
echo " MTP ${MTP_DESC} / CUDA GRAPH OFF"
echo '============================================================'

bash "$ROOT_DIR/qwen38_vanilla_pp3_nvfp4_final_gate.sh"

echo
echo '=== LOAD / MEMORY / MTP SUMMARY ==='
docker logs "$CONTAINER" 2>&1 | \
  grep -Ei \
  'MTP|speculative|Qwen3_5ForCausalLMMTP|Load weight|KV Cache|Mamba Cache|Memory pool end|max_total_num_tokens|available_gpu_mem' | \
  tail -240 || true

if [[ "$RUN_LONG_STRESS" == "1" ]]; then
    echo
    echo '=== REAL PARALLEL-3 LONG-CONTEXT STRESS ==='
    bash "$ROOT_DIR/qwen38_vanilla_pp3_nvfp4_long_stress_gate.sh"
else
    echo
    echo '=== LONG STRESS SKIPPED ==='
fi

echo
echo 'QWEN38_GITTENSOR_PP3_NVFP4_PROBE=PASS'
