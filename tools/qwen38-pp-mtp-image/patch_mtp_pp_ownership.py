from pathlib import Path

EAGLE = Path("/sgl-workspace/sglang/python/sglang/srt/speculative/eagle_worker_v2.py")
MTP = Path("/sgl-workspace/sglang/python/sglang/srt/models/qwen3_5_mtp.py")

# 1) Non-draft PP ranks must never enter a speculative phase.
# Prefill-side PP constructs the native MTP draft only on the last stage.
s = EAGLE.read_text()

phase_needles = [
    '''            # Draft prefill\n            with (\n                self.draft_worker.draft_tp_context(\n                    self.draft_worker.draft_runner.tp_group\n                ),\n''',
    '''                with (\n                    self.draft_worker.draft_tp_context(\n                        self.draft_worker.draft_runner.tp_group\n                    ),\n                    speculative_moe_backend_context(),\n                    speculative_moe_a2a_backend_context(),\n                    spec_stage_span("draft"),\n''',
    '''                with (\n                    self.draft_worker.draft_tp_context(\n                        self.draft_worker.draft_runner.tp_group\n                    ),\n                    speculative_moe_backend_context(),\n                    speculative_moe_a2a_backend_context(),\n                    spec_stage_span("draft_extend"),\n''',
]

phase_replacements = [
    '''            # Prefill-side PP constructs the native MTP draft only on the\n            # last stage. Earlier PP ranks publish/relay the target result and\n            # must not dereference a missing draft worker.\n            if self._draft_worker is None:\n                return batch_output\n\n            # Draft prefill\n            with (\n                self.draft_worker.draft_tp_context(\n                    self.draft_worker.draft_runner.tp_group\n                ),\n''',
    '''                if self._draft_worker is None:\n                    return batch_output\n                with (\n                    self.draft_worker.draft_tp_context(\n                        self.draft_worker.draft_runner.tp_group\n                    ),\n                    speculative_moe_backend_context(),\n                    speculative_moe_a2a_backend_context(),\n                    spec_stage_span("draft"),\n''',
    '''                if self._draft_worker is None:\n                    return batch_output\n                with (\n                    self.draft_worker.draft_tp_context(\n                        self.draft_worker.draft_runner.tp_group\n                    ),\n                    speculative_moe_backend_context(),\n                    speculative_moe_a2a_backend_context(),\n                    spec_stage_span("draft_extend"),\n''',
]

for needle, replacement in zip(phase_needles, phase_replacements):
    if needle in s:
        s = s.replace(needle, replacement)

EAGLE.write_text(s)

# 2) A non-owning PP target stage can expose a placeholder/empty endpoint
# parameter rather than None. Never replace the draft's self-contained full
# embedding/head with such a placeholder.
s = MTP.read_text()
old = '''        if embed is not None:\n            del self.model.embed_tokens.weight\n            self.model.embed_tokens.weight = embed\n        if head is not None and not self.config.tie_word_embeddings:\n            del self.lm_head.weight\n            self.lm_head.weight = head\n'''
new = '''        def _is_real_endpoint(weight):\n            return (\n                isinstance(weight, torch.Tensor)\n                and weight.numel() > 0\n                and weight.ndim >= 2\n            )\n\n        if _is_real_endpoint(embed):\n            del self.model.embed_tokens.weight\n            self.model.embed_tokens.weight = embed\n        else:\n            logger.info(\n                "[MTP-PP-ENDPOINT] keeping draft embedding; target PP stage "\n                "does not own a real embedding weight"\n            )\n        if (\n            _is_real_endpoint(head)\n            and not self.config.tie_word_embeddings\n        ):\n            del self.lm_head.weight\n            self.lm_head.weight = head\n        elif not self.config.tie_word_embeddings:\n            logger.info(\n                "[MTP-PP-ENDPOINT] keeping draft lm_head; target PP stage "\n                "does not own a real lm_head weight"\n            )\n'''

if "[MTP-PP-ENDPOINT] keeping draft embedding" not in s:
    if old not in s:
        raise RuntimeError("Qwen3.5 MTP endpoint sharing block not found")
    s = s.replace(old, new, 1)

# 3) Fail before CUDA embedding lookup if PP handed the draft non-vocabulary
# ids. This distinguishes MM pad/proxy ids from an incorrectly shaped MTP
# embedding table without poisoning the CUDA context with a device assert.
old_embed = '''            if input_embeds is None:\n                input_embeds = self.model.embed_tokens(input_ids)\n\n            hidden_states = forward_batch.spec_info.hidden_states\n'''
new_embed = '''            if input_embeds is None:\n                _embed_weight = self.model.embed_tokens.weight\n                _rows = int(_embed_weight.shape[0])\n                if input_ids.numel() > 0:\n                    _min_id = int(input_ids.min().item())\n                    _max_id = int(input_ids.max().item())\n                else:\n                    _min_id = 0\n                    _max_id = -1\n                if _min_id < 0 or _max_id >= _rows:\n                    raise RuntimeError(\n                        "[MTP-PP-INPUT-OOB] "\n                        f"input_ids=[{_min_id},{_max_id}] "\n                        f"embed_rows={_rows} "\n                        f"config_vocab={self.config.vocab_size} "\n                        f"mode={forward_batch.forward_mode}"\n                    )\n                logger.info(\n                    "[MTP-PP-INPUT-RANGE] input_ids=[%d,%d] "\n                    "embed_rows=%d config_vocab=%d",\n                    _min_id, _max_id, _rows, int(self.config.vocab_size),\n                )\n                input_embeds = self.model.embed_tokens(input_ids)\n\n            hidden_states = forward_batch.spec_info.hidden_states\n'''

if "[MTP-PP-INPUT-OOB]" not in s:
    if old_embed not in s:
        raise RuntimeError("Qwen3.5 MTP token embedding block not found")
    s = s.replace(old_embed, new_embed, 1)

MTP.write_text(s)
print("PATCHED native MTP PP ownership, endpoint sharing, and input-id preflight")
print(EAGLE)
print(MTP)
