FROM lmsysorg/sglang:qwen38-27b

# Production baseline: the validated PP3+MTP path no longer needs synchronous
# CUDA error attribution.  Keep launch blocking explicitly off for throughput.
ENV CUDA_LAUNCH_BLOCKING=0

COPY tools/qwen38-pp-mtp-image/patch_pp_spec_guard.py /tmp/patch_pp_spec_guard.py
COPY tools/qwen38-pp-mtp-image/patch_mtp_early_share.py /tmp/patch_mtp_early_share.py
COPY tools/qwen38-pp-mtp-image/patch_mtp_pp_prefill_input.py /tmp/patch_mtp_pp_prefill_input.py
COPY tools/qwen38-pp-mtp-image/patch_mtp_pp_mm_fallback.py /tmp/patch_mtp_pp_mm_fallback.py
COPY tools/qwen38-pp-mtp-image/patch_mtp_pp_ownership.py /tmp/patch_mtp_pp_ownership.py
COPY tools/qwen38-pp-mtp-image/patch_mtp_pp_cuda_graph_safe_input_guard.py /tmp/patch_mtp_pp_cuda_graph_safe_input_guard.py
COPY tools/qwen38-pp-mtp-image/patch_mtp_pp_req_source.py /tmp/patch_mtp_pp_req_source.py
COPY tools/qwen38-pp-mtp-image/patch_mtp_pp_mm_identity.py /tmp/patch_mtp_pp_mm_identity.py
COPY tools/qwen38-pp-mtp-image/patch_mtp_pp_spec_bridge.py /tmp/patch_mtp_pp_spec_bridge.py
COPY tools/qwen38-pp-mtp-image/patch_mtp_pp_prefill_bridge_final_chunk.py /tmp/patch_mtp_pp_prefill_bridge_final_chunk.py
COPY tools/qwen38-pp-mtp-image/fix_mtp_pp_spec_bridge_installer.py /tmp/fix_mtp_pp_spec_bridge_installer.py
COPY tools/qwen38-pp-mtp-image/patch_mtp_pp_bridge_serializer.py /tmp/patch_mtp_pp_bridge_serializer.py
COPY tools/qwen38-pp-mtp-image/patch_mtp_pp_verify_nvfp4_metadata.py /tmp/patch_mtp_pp_verify_nvfp4_metadata.py
COPY tools/qwen38-pp-mtp-image/patch_mtp_pp_pool_profile.py /tmp/patch_mtp_pp_pool_profile.py
COPY tools/qwen38-pp-mtp-image/patch_mtp_pp_local_memory_budget.py /tmp/patch_mtp_pp_local_memory_budget.py
COPY tools/qwen38-pp-mtp-image/patch_mtp_pp_embed_fp8.py /tmp/patch_mtp_pp_embed_fp8.py
COPY tools/qwen38-pp-mtp-image/patch_nvfp4_sequential_dequant.py /tmp/patch_nvfp4_sequential_dequant.py
COPY tools/qwen38-pp-mtp-image/patch_mtp_pp_sparse_mm_bridge.py /tmp/patch_mtp_pp_sparse_mm_bridge.py
COPY tools/qwen38-pp-mtp-image/patch_mtp_pp_graph_scope.py /tmp/patch_mtp_pp_graph_scope.py
COPY tools/qwen38-pp-mtp-image/patch_mtp_pp_quiet_logs.py /tmp/patch_mtp_pp_quiet_logs.py

RUN python3 /tmp/patch_pp_spec_guard.py \
 && python3 /tmp/patch_mtp_early_share.py \
 && python3 /tmp/patch_mtp_pp_prefill_input.py \
 && python3 /tmp/patch_mtp_pp_mm_fallback.py \
 && python3 /tmp/patch_mtp_pp_ownership.py \
 && python3 /tmp/patch_mtp_pp_cuda_graph_safe_input_guard.py \
 && python3 /tmp/patch_mtp_pp_req_source.py \
 && python3 /tmp/patch_mtp_pp_mm_identity.py \
 && python3 /tmp/fix_mtp_pp_spec_bridge_installer.py \
 && python3 /tmp/patch_mtp_pp_spec_bridge.py \
 && python3 /tmp/patch_mtp_pp_prefill_bridge_final_chunk.py \
 && python3 /tmp/patch_mtp_pp_bridge_serializer.py \
 && python3 /tmp/patch_mtp_pp_verify_nvfp4_metadata.py \
 && python3 /tmp/patch_mtp_pp_pool_profile.py \
 && python3 /tmp/patch_mtp_pp_local_memory_budget.py \
 && python3 /tmp/patch_mtp_pp_embed_fp8.py \
 && python3 /tmp/patch_nvfp4_sequential_dequant.py \
 && python3 /tmp/patch_mtp_pp_sparse_mm_bridge.py \
 && python3 /tmp/patch_mtp_pp_graph_scope.py \
 && python3 /tmp/patch_mtp_pp_quiet_logs.py \
 && python3 -m py_compile \
      /sgl-workspace/sglang/python/sglang/srt/server_args.py \
      /sgl-workspace/sglang/python/sglang/srt/managers/scheduler.py \
      /sgl-workspace/sglang/python/sglang/srt/managers/scheduler_pp_mixin.py \
      /sgl-workspace/sglang/python/sglang/srt/managers/mm_utils.py \
      /sgl-workspace/sglang/python/sglang/srt/mem_cache/kv_cache_configurator.py \
      /sgl-workspace/sglang/python/sglang/srt/layers/logits_processor.py \
      /sgl-workspace/sglang/python/sglang/srt/layers/quantization/fp4_kv_cache_quant_method.py \
      /sgl-workspace/sglang/python/sglang/srt/speculative/eagle_worker_common.py \
      /sgl-workspace/sglang/python/sglang/srt/speculative/eagle_worker_v2.py \
      /sgl-workspace/sglang/python/sglang/srt/models/qwen3_5.py \
      /sgl-workspace/sglang/python/sglang/srt/models/qwen3_5_mtp.py \
 && rm \
      /tmp/patch_pp_spec_guard.py \
      /tmp/patch_mtp_early_share.py \
      /tmp/patch_mtp_pp_prefill_input.py \
      /tmp/patch_mtp_pp_mm_fallback.py \
      /tmp/patch_mtp_pp_ownership.py \
      /tmp/patch_mtp_pp_cuda_graph_safe_input_guard.py \
      /tmp/patch_mtp_pp_req_source.py \
      /tmp/patch_mtp_pp_mm_identity.py \
      /tmp/patch_mtp_pp_spec_bridge.py \
      /tmp/patch_mtp_pp_prefill_bridge_final_chunk.py \
      /tmp/fix_mtp_pp_spec_bridge_installer.py \
      /tmp/patch_mtp_pp_bridge_serializer.py \
      /tmp/patch_mtp_pp_verify_nvfp4_metadata.py \
      /tmp/patch_mtp_pp_pool_profile.py \
      /tmp/patch_mtp_pp_local_memory_budget.py \
      /tmp/patch_mtp_pp_embed_fp8.py \
      /tmp/patch_nvfp4_sequential_dequant.py \
      /tmp/patch_mtp_pp_sparse_mm_bridge.py \
      /tmp/patch_mtp_pp_graph_scope.py \
      /tmp/patch_mtp_pp_quiet_logs.py
