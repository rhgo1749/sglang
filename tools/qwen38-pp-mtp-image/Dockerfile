FROM lmsysorg/sglang:qwen38-27b

# Diagnostic image: make the first failing CUDA op report at the real Python
# call site instead of surfacing later at an unrelated synchronization point.
ENV CUDA_LAUNCH_BLOCKING=1

COPY tools/qwen38-pp-mtp-image/patch_pp_spec_guard.py /tmp/patch_pp_spec_guard.py
COPY tools/qwen38-pp-mtp-image/patch_mtp_early_share.py /tmp/patch_mtp_early_share.py
COPY tools/qwen38-pp-mtp-image/patch_mtp_pp_prefill_input.py /tmp/patch_mtp_pp_prefill_input.py
COPY tools/qwen38-pp-mtp-image/patch_mtp_pp_mm_fallback.py /tmp/patch_mtp_pp_mm_fallback.py
COPY tools/qwen38-pp-mtp-image/patch_mtp_pp_ownership.py /tmp/patch_mtp_pp_ownership.py
COPY tools/qwen38-pp-mtp-image/patch_mtp_pp_req_source.py /tmp/patch_mtp_pp_req_source.py

RUN python3 /tmp/patch_pp_spec_guard.py \
 && python3 /tmp/patch_mtp_early_share.py \
 && python3 /tmp/patch_mtp_pp_prefill_input.py \
 && python3 /tmp/patch_mtp_pp_mm_fallback.py \
 && python3 /tmp/patch_mtp_pp_ownership.py \
 && python3 /tmp/patch_mtp_pp_req_source.py \
 && python3 -m py_compile \
      /sgl-workspace/sglang/python/sglang/srt/server_args.py \
      /sgl-workspace/sglang/python/sglang/srt/managers/scheduler_pp_mixin.py \
      /sgl-workspace/sglang/python/sglang/srt/speculative/eagle_worker_v2.py \
      /sgl-workspace/sglang/python/sglang/srt/models/qwen3_5_mtp.py \
 && rm \
      /tmp/patch_pp_spec_guard.py \
      /tmp/patch_mtp_early_share.py \
      /tmp/patch_mtp_pp_prefill_input.py \
      /tmp/patch_mtp_pp_mm_fallback.py \
      /tmp/patch_mtp_pp_ownership.py \
      /tmp/patch_mtp_pp_req_source.py
