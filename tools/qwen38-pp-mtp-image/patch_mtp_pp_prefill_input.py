from pathlib import Path

MTP = Path(
    "/sgl-workspace/sglang/python/sglang/srt/models/qwen3_5_mtp.py"
)

s = MTP.read_text()

old = '''        try:
            assert input_embeds is None
            input_embeds = forward_batch.mm_input_embeds
'''

new = '''        try:
            # PP compatibility: a pipeline stage may already provide the input
            # embeddings needed by native MTP prefill.  The original colocated
            # path expected input_embeds=None and sourced only multimodal embeds
            # from ForwardBatch, which crashes as soon as PP forwards a tensor.
            # Prefer the explicit pipeline-provided tensor; preserve the original
            # multimodal fallback when no tensor was supplied.
            if input_embeds is None:
                input_embeds = forward_batch.mm_input_embeds
            else:
                logger.debug(
                    "[MTP-PP-PREFILL-INPUT] using pipeline-provided input_embeds shape=%s",
                    tuple(input_embeds.shape),
                )
'''

if "[MTP-PP-PREFILL-INPUT]" not in s:
    if old not in s:
        raise RuntimeError(
            "Qwen3_5ForCausalLMMTP forward input_embeds guard not found"
        )
    s = s.replace(old, new, 1)

MTP.write_text(s)
print("PATCHED native MTP PP prefill input handling")
print(MTP)
