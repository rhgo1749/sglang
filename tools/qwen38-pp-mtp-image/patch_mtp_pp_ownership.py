from pathlib import Path

EAGLE = Path("/sgl-workspace/sglang/python/sglang/srt/speculative/eagle_worker_v2.py")
MTP = Path("/sgl-workspace/sglang/python/sglang/srt/models/qwen3_5_mtp.py")

# 1) Non-draft PP ranks must stop after publishing the target result.
# The last PP stage owns the colocated MTP draft; earlier stages are relay/target-only.
s = EAGLE.read_text()
needle = '''            # Draft prefill
            with (
                self.draft_worker.draft_tp_context(
                    self.draft_worker.draft_runner.tp_group
                ),
'''
replacement = '''            # Prefill-side PP constructs the native MTP draft only on the
            # last stage. Earlier PP ranks publish/relay the target result and
            # must not dereference a missing draft worker.
            if self._draft_worker is None:
                return batch_output

            # Draft prefill
            with (
                self.draft_worker.draft_tp_context(
                    self.draft_worker.draft_runner.tp_group
                ),
'''

if "must not dereference a missing draft worker" not in s:
    count = s.count(needle)
    if count == 0:
        raise RuntimeError("native MTP draft-prefill context block not found")
    # There are normally two symmetric prefill paths. Patch all exact matches.
    s = s.replace(needle, replacement)

EAGLE.write_text(s)

# 2) A non-owning PP target stage can expose a placeholder/empty endpoint
# parameter rather than None. Never replace the draft's self-contained full
# embedding/head with such a placeholder.
s = MTP.read_text()
old = '''        if embed is not None:
            del self.model.embed_tokens.weight
            self.model.embed_tokens.weight = embed
        if head is not None and not self.config.tie_word_embeddings:
            del self.lm_head.weight
            self.lm_head.weight = head
'''
new = '''        def _is_real_endpoint(weight):
            return (
                isinstance(weight, torch.Tensor)
                and weight.numel() > 0
                and weight.ndim >= 2
            )

        if _is_real_endpoint(embed):
            del self.model.embed_tokens.weight
            self.model.embed_tokens.weight = embed
        else:
            logger.info(
                "[MTP-PP-ENDPOINT] keeping draft embedding; target PP stage "
                "does not own a real embedding weight"
            )
        if (
            _is_real_endpoint(head)
            and not self.config.tie_word_embeddings
        ):
            del self.lm_head.weight
            self.lm_head.weight = head
        elif not self.config.tie_word_embeddings:
            logger.info(
                "[MTP-PP-ENDPOINT] keeping draft lm_head; target PP stage "
                "does not own a real lm_head weight"
            )
'''

if "[MTP-PP-ENDPOINT] keeping draft embedding" not in s:
    if old not in s:
        raise RuntimeError("Qwen3.5 MTP endpoint sharing block not found")
    s = s.replace(old, new, 1)

MTP.write_text(s)
print("PATCHED native MTP PP draft ownership and endpoint sharing")
print(EAGLE)
print(MTP)
