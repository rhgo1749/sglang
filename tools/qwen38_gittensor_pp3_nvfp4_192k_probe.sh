#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export MODEL="${MODEL:-gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090}"
export CONTAINER="${CONTAINER:-sglang-qwen38-gittensor-pp3}"

export CTX="${CTX:-196608}"
export CONTEXT_LENGTH="${CONTEXT_LENGTH:-196608}"

# Native MTP + row-wise FP8 embedding candidate.
# 22,28,14 moves only target layer 22 from PP0 to PP1 relative to 23,27,14.
# Layer 22 is linear-attention, so the full-attention distribution stays 5/7/4
# and KV bytes/token do not increase.  At mem_fraction_static=0.84 this split
# boots and functions correctly but stops at 572608 tokens, only 17216 below
# the exact 3x196608 target. Keep the partition fixed and recover that final
# budget from static slack instead of pushing another target layer onto PP2.
export PARTITION="${PARTITION:-22,28,14}"
# 0.855 reclaims roughly 0.22 GiB of the ~15 GiB pre-load slack versus 0.84.
# The exact token cap below prevents that extra budget from growing the pool
# beyond 3x196608; it is only enough to let the remaining PP-local bottleneck
# reach the requested capacity while preserving PP2 post-pool workspace.
export MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.855}"
# Exact 3 x 196608 target. Keep the old extra 4096-token reserve disabled while
# validating the native MTP capacity and post-pool backend headroom.
export MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-589824}"

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

bash "$ROOT_DIR/qwen38_failfast_watch.sh" \
  bash "$ROOT_DIR/qwen38_vanilla_pp3_nvfp4_final_gate.sh"

echo
echo '=== MTP PP CAPACITY PROFILE ==='
docker logs "$CONTAINER" 2>&1 | \
  grep -E '\[MTP-PP-(LOCAL-MEM-BUDGET|CAPACITY-(LOCAL|GLOBAL))\]' || true

echo
echo '=== MTP PHYSICAL MEMORY AUDIT ==='
docker logs "$CONTAINER" 2>&1 | \
  grep -E '\[MTP-(EARLY-SHARE|PP-FP8-EMBED|MEM-AUDIT|MEM-TOP)\]' || true

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
