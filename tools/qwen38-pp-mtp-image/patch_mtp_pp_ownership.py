from pathlib import Path

EAGLE = Path("/sgl-workspace/sglang/python/sglang/srt/speculative/eagle_worker_v2.py")
MTP = Path("/sgl-workspace/sglang/python/sglang/srt/models/qwen3_5_mtp.py")

# 1) Prefill-side PP constructs the native MTP draft only on the last stage.
# Return from earlier PP ranks only AFTER target prefill has produced batch_output.
s = EAGLE.read_text()

safe_prefill_needle = '''            # Publish before draft_extend so the fence is at target-end.\n            if on_publish is not None:\n                on_publish(batch_output.new_seq_lens)\n\n            # Draft prefill\n'''
safe_prefill_replacement = '''            # Publish before draft_extend so the fence is at target-end.\n            if on_publish is not None:\n                on_publish(batch_output.new_seq_lens)\n\n            # Prefill-side PP owns the native MTP draft only on the last stage.\n            # Earlier stages have a valid target batch_output but no draft worker.\n            if self._draft_worker is None:\n                return batch_output\n\n            # Draft prefill\n'''
if "Earlier stages have a valid target batch_output but no draft worker" not in s:
    if safe_prefill_needle not in s:
        raise RuntimeError("safe native MTP PP prefill insertion point not found")
    s = s.replace(safe_prefill_needle, safe_prefill_replacement, 1)

# 2) The last PP stage can receive proxy/placeholder values in ScheduleBatch.input_ids
# because target execution there consumes pipeline hidden states, not token embeddings.
# Native MTP *does* need real token ids, so rebuild the current extend token stream from
# the replicated Req metadata before applying EAGLE's one-token left rotation.
construct_needle = '''        # Construct input_ids\n        if not batch.forward_mode.is_idle():\n            # Chunked-prefill-aware tail tokens (see PR #26329).\n            tail_tokens = _eagle_prefill_tail_tokens(batch, next_token_ids)\n'''
construct_replacement = '''        # Construct input_ids\n        if not batch.forward_mode.is_idle():\n            _vocab_size = int(self.target_worker.model_config.vocab_size)\n\n            # PP relay stages may carry non-token proxy values in batch.input_ids.\n            # For text-only draft prefill, reconstruct the exact extend token stream\n            # the scheduler originally built from Req metadata.\n            if batch.input_ids.numel() > 0:\n                _raw_min = int(batch.input_ids.min().item())\n                _raw_max = int(batch.input_ids.max().item())\n            else:\n                _raw_min, _raw_max = 0, -1\n\n            if (_raw_min < 0 or _raw_max >= _vocab_size) and mm_input_embeds is None:\n                _restored = []\n                for _req, _extend_len in zip(batch.reqs, batch.extend_lens):\n                    _extend_len = int(_extend_len)\n                    _fill_ids = _req.get_fill_ids()\n                    _start = len(_req.prefix_indices)\n                    _piece = _fill_ids[_start : _start + _extend_len]\n                    if len(_piece) != _extend_len:\n                        raise RuntimeError(\n                            "[MTP-PP-RESTORE-LEN] "\n                            f"rid={_req.rid} start={_start} "\n                            f"need={_extend_len} got={len(_piece)}"\n                        )\n                    _restored.extend(int(_x) for _x in _piece)\n\n                if len(_restored) != int(batch.input_ids.numel()):\n                    raise RuntimeError(\n                        "[MTP-PP-RESTORE-TOTAL] "\n                        f"restored={len(_restored)} "\n                        f"batch_input={int(batch.input_ids.numel())}"\n                    )\n\n                batch.input_ids = torch.tensor(\n                    _restored,\n                    dtype=batch.input_ids.dtype,\n                    device=batch.input_ids.device,\n                )\n                _new_min = int(batch.input_ids.min().item()) if batch.input_ids.numel() else 0\n                _new_max = int(batch.input_ids.max().item()) if batch.input_ids.numel() else -1\n                logger.info(\n                    "[MTP-PP-RESTORE-IDS] raw=[%d,%d] restored=[%d,%d] tokens=%d",\n                    _raw_min, _raw_max, _new_min, _new_max, int(batch.input_ids.numel()),\n                )\n                if _new_min < 0 or _new_max >= _vocab_size:\n                    raise RuntimeError(\n                        "[MTP-PP-RESTORE-OOB] "\n                        f"restored=[{_new_min},{_new_max}] vocab={_vocab_size}"\n                    )\n\n            # The final rotated token comes from the target model. Validate it\n            # separately so a PP bonus-token handoff bug is distinguishable from\n            # corrupted prompt ids.\n            if next_token_ids.numel() > 0:\n                _next_min = int(next_token_ids.min().item())\n                _next_max = int(next_token_ids.max().item())\n                logger.info(\n                    "[MTP-PP-NEXT-TOKEN-RANGE] next=[%d,%d] vocab=%d",\n                    _next_min, _next_max, _vocab_size,\n                )\n                if _next_min < 0 or _next_max >= _vocab_size:\n                    raise RuntimeError(\n                        "[MTP-PP-NEXT-TOKEN-OOB] "\n                        f"next=[{_next_min},{_next_max}] vocab={_vocab_size}"\n                    )\n\n            # Chunked-prefill-aware tail tokens (see PR #26329).\n            tail_tokens = _eagle_prefill_tail_tokens(batch, next_token_ids)\n'''
if "[MTP-PP-RESTORE-IDS]" not in s:
    if construct_needle not in s:
        raise RuntimeError("native MTP draft prefill input construction block not found")
    s = s.replace(construct_needle, construct_replacement, 1)

EAGLE.write_text(s)

# 3) A non-owning PP target stage can expose a placeholder/empty endpoint
# parameter rather than None. Never replace the draft's self-contained full
# embedding/head with such a placeholder.
s = MTP.read_text()
old = '''        if embed is not None:\n            del self.model.embed_tokens.weight\n            self.model.embed_tokens.weight = embed\n        if head is not None and not self.config.tie_word_embeddings:\n            del self.lm_head.weight\n            self.lm_head.weight = head\n'''
new = '''        def _is_real_endpoint(weight):\n            return (\n                isinstance(weight, torch.Tensor)\n                and weight.numel() > 0\n                and weight.ndim >= 2\n            )\n\n        if _is_real_endpoint(embed):\n            del self.model.embed_tokens.weight\n            self.model.embed_tokens.weight = embed\n        else:\n            logger.info(\n                "[MTP-PP-ENDPOINT] keeping draft embedding; target PP stage "\n                "does not own a real embedding weight"\n            )\n        if (\n            _is_real_endpoint(head)\n            and not self.config.tie_word_embeddings\n        ):\n            del self.lm_head.weight\n            self.lm_head.weight = head\n        elif not self.config.tie_word_embeddings:\n            logger.info(\n                "[MTP-PP-ENDPOINT] keeping draft lm_head; target PP stage "\n                "does not own a real lm_head weight"\n            )\n'''

if "[MTP-PP-ENDPOINT] keeping draft embedding" not in s:
    if old not in s:
        raise RuntimeError("Qwen3.5 MTP endpoint sharing block not found")
    s = s.replace(old, new, 1)

# 4) Keep a final preflight at the actual embedding lookup. This should now
# remain quiet unless a later code path corrupts the reconstructed ids.
old_embed = '''            if input_embeds is None:\n                input_embeds = self.model.embed_tokens(input_ids)\n\n            hidden_states = forward_batch.spec_info.hidden_states\n'''
new_embed = '''            if input_embeds is None:\n                _embed_weight = self.model.embed_tokens.weight\n                _rows = int(_embed_weight.shape[0])\n                if input_ids.numel() > 0:\n                    _min_id = int(input_ids.min().item())\n                    _max_id = int(input_ids.max().item())\n                else:\n                    _min_id = 0\n                    _max_id = -1\n                if _min_id < 0 or _max_id >= _rows:\n                    raise RuntimeError(\n                        "[MTP-PP-INPUT-OOB] "\n                        f"input_ids=[{_min_id},{_max_id}] "\n                        f"embed_rows={_rows} "\n                        f"config_vocab={self.config.vocab_size} "\n                        f"mode={forward_batch.forward_mode}"\n                    )\n                logger.info(\n                    "[MTP-PP-INPUT-RANGE] input_ids=[%d,%d] "\n                    "embed_rows=%d config_vocab=%d",\n                    _min_id, _max_id, _rows, int(self.config.vocab_size),\n                )\n                input_embeds = self.model.embed_tokens(input_ids)\n\n            hidden_states = forward_batch.spec_info.hidden_states\n'''

if "[MTP-PP-INPUT-OOB]" not in s:
    if old_embed not in s:
        raise RuntimeError("Qwen3.5 MTP token embedding block not found")
    s = s.replace(old_embed, new_embed, 1)

MTP.write_text(s)
print("PATCHED native MTP PP safe ownership, prompt-id restoration, and endpoint sharing")
print(EAGLE)
print(MTP)
