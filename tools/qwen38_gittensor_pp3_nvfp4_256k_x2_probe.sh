#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export IMAGE="${IMAGE:-sglang:qwen38-27b-pp-mtp-share}"
export MODEL="${MODEL:-gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090}"
export CONTAINER="${CONTAINER:-sglang-qwen38-gittensor-pp3}"
export PORT="${PORT:-30001}"

export CTX="${CTX:-262144}"
export CONTEXT_LENGTH="${CONTEXT_LENGTH:-262144}"
export PARALLEL="${PARALLEL:-2}"
export MAX_RUNNING="${MAX_RUNNING:-2}"
export MAX_MAMBA="${MAX_MAMBA:-2}"

# 2x256K reaches the exact 524288-token pool with 22,28,14, but PP2 still
# OOMs during draft NVFP4 dequantization: a 106 MiB allocation sees only
# ~100 MiB free around 109K prefill. Move layers 22-23 onto PP0 and layers
# 50-51 onto PP1, yielding 24,28,12. With full-attention every fourth layer,
# the target full-attention distribution changes from 5/7/4 to 6/7/3. This
# removes one full-attention KV layer plus two target layers from PP2 without
# increasing PP1's full-attention count, preserving the exact 2x256K pool.
export PARTITION="${PARTITION:-24,28,12}"
# Keep the validated 2x256K memory policy; this probe changes only PP placement.
export MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.84}"
export MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-524288}"
export CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-512}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

export ENABLE_MTP="${ENABLE_MTP:-1}"
export SPECULATIVE_ALGO="${SPECULATIVE_ALGO:-NEXTN}"
export SPECULATIVE_NUM_STEPS="${SPECULATIVE_NUM_STEPS:-3}"
export SPECULATIVE_EAGLE_TOPK="${SPECULATIVE_EAGLE_TOPK:-1}"
export SPECULATIVE_NUM_DRAFT_TOKENS="${SPECULATIVE_NUM_DRAFT_TOKENS:-4}"

# Exercise almost the entire 262144-token window while leaving room for the
# requested decode and a little boundary slack. Capacity itself is still gated
# at the full 2 x 262144 = 524288 tokens.
export STAGE="${STAGE:-262000}"
export DECODE_TOKENS="${DECODE_TOKENS:-8}"
export REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-1200}"

echo '============================================================'
echo ' QWEN3.8 GITTENSOR NVFP4 PP3 / NATIVE MTP / 2x256K PROBE'
echo " model=${MODEL}"
echo " context_length=${CTX}"
echo " parallel=${PARALLEL}"
echo " required_capacity=$((CTX * MAX_RUNNING))"
echo " partition=${PARTITION}"
echo " mem_fraction_static=${MEM_FRACTION_STATIC}"
echo " max_total_tokens=${MAX_TOTAL_TOKENS}"
echo " chunked_prefill_size=${CHUNKED_PREFILL_SIZE}"
echo " max_running_requests=${MAX_RUNNING}"
echo " max_mamba_cache_size=${MAX_MAMBA}"
echo " stress_prompt_tokens=${STAGE}"
echo " decode_tokens=${DECODE_TOKENS}"
echo ' MTP ON / CUDA GRAPH OFF'
echo '============================================================'

bash "$ROOT_DIR/qwen38_failfast_watch.sh" \
  bash "$ROOT_DIR/qwen38_gittensor_pp3_nvfp4_256k_x2_gate.sh"

echo
echo '=== MTP PP CAPACITY PROFILE ==='
docker logs "$CONTAINER" 2>&1 | \
  grep -E '\[MTP-PP-(LOCAL-MEM-BUDGET|CAPACITY-(LOCAL|GLOBAL))\]' || true

echo
echo '=== MTP PHYSICAL MEMORY AUDIT ==='
docker logs "$CONTAINER" 2>&1 | \
  grep -E '\[MTP-(EARLY-SHARE|PP-FP8-EMBED|MEM-AUDIT|MEM-TOP)\]' || true

echo
echo 'QWEN38_GITTENSOR_PP3_NVFP4_X2_256K_PROBE=PASS'
