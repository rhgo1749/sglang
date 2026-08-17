from pathlib import Path

MTP = Path(
    "/sgl-workspace/sglang/python/sglang/srt/models/qwen3_5_mtp.py"
)

s = MTP.read_text()

old = '''            if (
                forward_batch.forward_mode.is_extend()
                and forward_batch.contains_mm_inputs()
                and not forward_batch.forward_mode.is_draft_extend_v2()
            ):
                assert input_embeds is not None
                last_indices = (
                    forward_batch.extend_start_loc + forward_batch.extend_seq_lens - 1
                ).long()
                input_embeds[last_indices] = self.model.embed_tokens(
                    input_ids[last_indices]
                )
'''

new = '''            if (
                forward_batch.forward_mode.is_extend()
                and forward_batch.contains_mm_inputs()
                and not forward_batch.forward_mode.is_draft_extend_v2()
            ):
                # PP native-MTP may inherit the multimodal-capable ForwardBatch
                # marker even for a text-only request while no MM embedding tensor
                # is present on the draft stage.  In that case fall through to the
                # normal token-embedding path below instead of crashing here.
                if input_embeds is None:
                    logger.warning(
                        "[MTP-PP-MM-FALLBACK] contains_mm_inputs=True but no "
                        "input_embeds reached the draft stage; falling back to "
                        "token embeddings"
                    )
                else:
                    last_indices = (
                        forward_batch.extend_start_loc
                        + forward_batch.extend_seq_lens
                        - 1
                    ).long()
                    input_embeds[last_indices] = self.model.embed_tokens(
                        input_ids[last_indices]
                    )
'''

if "[MTP-PP-MM-FALLBACK]" not in s:
    if old not in s:
        raise RuntimeError("Qwen3.5 MTP multimodal input assertion block not found")
    s = s.replace(old, new, 1)

MTP.write_text(s)
print("PATCHED native MTP PP multimodal/text fallback")
print(MTP)
