#!/usr/bin/env bash
set -euo pipefail

REPO="${HOME}/projects/sglang-fork"
BASE="${REPO}/tools/qwen38_mtp_sidecar_parallel3_nvfp4_smoke.sh"
TMP="$(mktemp)"
trap 'rm -f "$TMP"' EXIT

cd "$REPO"
python3 -m py_compile tools/qwen38_mtp_cutover_hybrid_fp8_draft_hotfix.py

# Do not let a failed curl parse response JSON from a previous successful run.
rm -f /tmp/qwen38-nvfp4-p3-*.json \
      /tmp/qwen38-nvfp4-p3-*-http.txt \
      /tmp/qwen38-nvfp4-p3-runtime.log 2>/dev/null || true

# Reuse the mature parallel-3 harness, but replace the experimental
# NVFP4-draft/page1 pair with the production hybrid hotfix.  The target stays
# NVFP4 + page64; only CUDA2 becomes FP8 E4M3 + FlashInfer + page1.
sed \
  -e 's#tools/qwen38_mtp_cutover_nvfp4_target_hotfix.py#tools/qwen38_mtp_cutover_hybrid_fp8_draft_hotfix.py#g' \
  -e '/tools\/qwen38_mtp_cutover_sidecar_page1_hotfix.py/d' \
  -e 's/NVFP4 TARGET+DRAFT ABI/NVFP4 TARGET + FP8 DRAFT ABI/g' \
  -e 's/CUDA2 draft: NVFP4 KV (FlashInfer extend + TRTLLM-MHA multi-step decode, private page_size=1)/CUDA2 draft: FP8 E4M3 KV (FlashInfer multi-step decode + extend, private page_size=1)/g' \
  -e 's/HOTFIX NVFP4 TARGET \/ NVFP4 CUDA2 DRAFT/HOTFIX NVFP4 TARGET \/ FP8 CUDA2 DRAFT/g' \
  -e '/=== HOTFIX CUDA2 SIDECAR PAGE_SIZE=1 ===/d' \
  -e 's/NVFP4 TARGET KV, NVFP4 DRAFT KV/NVFP4 TARGET KV, FP8 DRAFT KV/g' \
  -e 's/target=nvfp4 CUDA2 draft=nvfp4/target=nvfp4 CUDA2 draft=fp8_e4m3/g' \
  -e 's/draft_kv_dtype=torch\\.float4_e2m1fn_x2\.\*draft_kv_tag=nvfp4/draft_kv_dtype=torch\\.float8_e4m3fn\.\*draft_kv_tag=fp8_e4m3/g' \
  -e 's/CUDA2 private NVFP4 draft-KV override did not activate/CUDA2 private FP8 draft-KV override did not activate/g' \
  -e 's/CUDA2 draft runner did not resolve NVFP4 KV/CUDA2 draft runner did not resolve FP8 E4M3 KV/g' \
  -e 's/TRTLLMHAAttnMultiStepDraftBackend\.\*extend=FlashInferAttnBackend/FlashInferMultiStepDraftBackend.*extend=FlashInferAttnBackend/g' \
  -e 's/CUDA2 draft did not resolve TRTLLM-MHA decode + FlashInfer extend split/CUDA2 draft did not resolve FlashInfer decode + extend/g' \
  -e 's/draft_kv=nvfp4/draft_kv=fp8_e4m3/g' \
  -e 's/MTP NVFP4-TARGET \/ NVFP4-DRAFT PARALLEL-3 SMOKE COMPLETE/MTP NVFP4-TARGET \/ FP8-DRAFT PARALLEL-3 SMOKE COMPLETE/g' \
  "$BASE" > "$TMP"

# The hybrid hotfix itself owns the CUDA2 page-size migration.  A stale page64
# gate in the generated harness would hide a bad transform, so fail before boot.
if grep -Eq 'draft=nvfp4|draft_kv=nvfp4|draft_page_size=64|TRTLLMHAAttnMultiStepDraftBackend.*extend=FlashInferAttnBackend' "$TMP"; then
  echo 'ERROR: generated hybrid smoke still contains experimental NVFP4-draft expectations'
  grep -nE 'draft=nvfp4|draft_kv=nvfp4|draft_page_size=64|TRTLLMHAAttnMultiStepDraftBackend' "$TMP" || true
  exit 1
fi

bash -n "$TMP"
echo 'hybrid smoke wrapper syntax: OK'
exec bash "$TMP"
