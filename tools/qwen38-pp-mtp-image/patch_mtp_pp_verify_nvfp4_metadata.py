from pathlib import Path

P = Path("/sgl-workspace/sglang/python/sglang/srt/speculative/eagle_worker_common.py")
s = P.read_text()

# TARGET_VERIFY intentionally goes through the extend attention kernels, but
# ForwardBatch.init_new treats it like decode for position setup and therefore
# leaves extend_{prefix,seq}_lens_cpu unset.  Quantized KV + FlashInfer extend
# dequantization still needs those host mirrors.  Reconstruct the exact verify
# semantics immediately after eagle_prepare_for_verify:
#   prefix length = committed target sequence length before the verify tree
#   extend length = verify tree width (draft_token_num) for each request.
marker = "[MTP-PP-VERIFY-NVFP4-LENS]"
if marker not in s:
    anchor = '''        verify_forward_batch, can_run_cuda_graph = eagle_prepare_for_verify(\n            verify_input,\n            req_to_token_pool,\n            batch,\n            target_worker,\n        )\n'''
    if anchor not in s:
        raise RuntimeError("run_eagle_verify prepare anchor not found")

    inject = '''\n        # [MTP-PP-VERIFY-NVFP4-LENS] TARGET_VERIFY uses extend attention, but\n        # ForwardBatch.init_new leaves the host extend-length mirrors unset.\n        # FlashInfer's quantized-KV dequant workspace requires them.\n        if verify_forward_batch.forward_mode.is_target_verify():\n            _prefix_cpu = verify_input.seq_lens_cpu\n            if _prefix_cpu is None:\n                _prefix_cpu = batch.seq_lens_cpu\n            if _prefix_cpu is None:\n                raise RuntimeError(\n                    "[MTP-PP-VERIFY-NVFP4-LENS] seq_lens_cpu missing for target verify"\n                )\n            if isinstance(_prefix_cpu, torch.Tensor):\n                _prefix_list = [int(x) for x in _prefix_cpu.tolist()]\n            else:\n                _prefix_list = [int(x) for x in _prefix_cpu]\n            verify_forward_batch.extend_prefix_lens_cpu = _prefix_list\n            verify_forward_batch.extend_seq_lens_cpu = [\n                int(verify_input.draft_token_num)\n            ] * len(_prefix_list)\n            print(\n                f"[MTP-PP-VERIFY-NVFP4-LENS] prefix={_prefix_list} "\n                f"extend={verify_forward_batch.extend_seq_lens_cpu}",\n                flush=True,\n            )\n'''
    s = s.replace(anchor, anchor + inject, 1)
    P.write_text(s)

# Semantic audit: require the target-verify reconstruction to live after the
# prepare call, not merely exist somewhere in the file.
s = P.read_text()
prepare_at = s.find("verify_forward_batch, can_run_cuda_graph = eagle_prepare_for_verify(")
marker_at = s.find(marker)
record_at = s.find("record_stream_each((batch.input_ids, batch.out_cache_loc)", prepare_at)
if prepare_at < 0 or marker_at < 0 or record_at < 0:
    raise RuntimeError("target-verify NVFP4 metadata audit anchors missing")
if not (prepare_at < marker_at < record_at):
    raise RuntimeError("target-verify NVFP4 metadata shim is not after prepare")
window = s[marker_at:record_at]
for required in (
    "extend_prefix_lens_cpu = _prefix_list",
    "extend_seq_lens_cpu = [",
    "verify_input.draft_token_num",
    "flush=True",
):
    if required not in window:
        raise RuntimeError(f"target-verify NVFP4 metadata shim missing: {required}")
if "logger.info(" in window:
    raise RuntimeError("target-verify NVFP4 metadata shim still depends on logger")

print("PATCHED TARGET_VERIFY NVFP4 FlashInfer host length metadata")
print("VERIFIED verify prefix/extend CPU mirrors after eagle_prepare_for_verify")
