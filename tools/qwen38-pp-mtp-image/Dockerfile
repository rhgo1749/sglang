FROM lmsysorg/sglang:qwen38-27b

COPY tools/qwen38-pp-mtp-image/patch_pp_spec_guard.py /tmp/patch_pp_spec_guard.py
COPY tools/qwen38-pp-mtp-image/patch_mtp_early_share.py /tmp/patch_mtp_early_share.py
COPY tools/qwen38-pp-mtp-image/patch_mtp_pp_prefill_input.py /tmp/patch_mtp_pp_prefill_input.py

RUN python3 /tmp/patch_pp_spec_guard.py \
 && python3 /tmp/patch_mtp_early_share.py \
 && python3 /tmp/patch_mtp_pp_prefill_input.py \
 && python3 -m py_compile \
      /sgl-workspace/sglang/python/sglang/srt/server_args.py \
      /sgl-workspace/sglang/python/sglang/srt/speculative/eagle_worker_v2.py \
      /sgl-workspace/sglang/python/sglang/srt/models/qwen3_5_mtp.py \
 && rm \
      /tmp/patch_pp_spec_guard.py \
      /tmp/patch_mtp_early_share.py \
      /tmp/patch_mtp_pp_prefill_input.py
