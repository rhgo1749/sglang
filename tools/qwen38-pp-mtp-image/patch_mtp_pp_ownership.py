from pathlib import Path

EAGLE = Path("/sgl-workspace/sglang/python/sglang/srt/speculative/eagle_worker_v2.py")
MTP = Path("/sgl-workspace/sglang/python/sglang/srt/models/qwen3_5_mtp.py")
SCHED = Path("/sgl-workspace/sglang/python/sglang/srt/managers/scheduler_pp_mixin.py")
KEY = "__mtp_pp_input_ids"

# ---------------------------------------------------------------------------
# 1) Carry the real token ids from PP0 through the PP proxy tensor transport.
# Later target stages may replace ScheduleBatch.input_ids with PP-local proxy
# values because they consume hidden states. Native MTP on the last PP stage
# still needs the original vocabulary ids.
# ---------------------------------------------------------------------------
s = SCHED.read_text()

launch_needle = '''                result = self.run_batch(cur_batch, pp_proxy_tensors)\n'''
launch_replacement = '''                # Native PP-MTP token side-channel. PP0 snapshots the real token\n                # ids before target execution; intermediate stages pop the side\n                # channel out of the model proxy and attach it to ScheduleBatch.\n                if self.pp_group.is_first_rank:\n                    if cur_batch.forward_mode.is_extend() or cur_batch.is_extend_in_batch:\n                        cur_batch._mtp_pp_input_ids = cur_batch.input_ids.detach().clone()\n                elif pp_proxy_tensors is not None:\n                    _mtp_ids = pp_proxy_tensors.tensors.pop("__mtp_pp_input_ids", None)\n                    if _mtp_ids is not None:\n                        cur_batch._mtp_pp_input_ids = _mtp_ids\n\n                result = self.run_batch(cur_batch, pp_proxy_tensors)\n'''
if "Native PP-MTP token side-channel" not in s:
    if launch_needle not in s:
        raise RuntimeError("PP launch insertion point not found")
    s = s.replace(launch_needle, launch_replacement, 1)

# Regular PP event loop: add transported ids to the outgoing proxy dict.
send_needle = '''                            self.send_proxy_work = self._pp_send_dict_to_next_stage(\n                                result.pp_hidden_states_proxy_tensors.tensors,\n                                async_send=True,\n                                msg_type="proxy",\n                            )\n'''
send_replacement = '''                            _mtp_proxy = result.pp_hidden_states_proxy_tensors.tensors\n                            _mtp_ids = getattr(cur_batch, "_mtp_pp_input_ids", None)\n                            if _mtp_ids is not None:\n                                _mtp_proxy["__mtp_pp_input_ids"] = _mtp_ids\n                            self.send_proxy_work = self._pp_send_dict_to_next_stage(\n                                _mtp_proxy,\n                                async_send=True,\n                                msg_type="proxy",\n                            )\n'''
if "_mtp_proxy = result.pp_hidden_states_proxy_tensors.tensors" not in s:
    if send_needle not in s:
        raise RuntimeError("regular PP proxy send insertion point not found")
    s = s.replace(send_needle, send_replacement, 1)

SCHED.write_text(s)

# ---------------------------------------------------------------------------
# 2) On the last PP stage, restore the transported ids before EAGLE rotates
# the prefill tokens. Do not attempt to reconstruct from Req metadata: the
# experiment proved that metadata is already PP-local/proxy-shaped there.
# ---------------------------------------------------------------------------
s = EAGLE.read_text()
construct_needle = '''        # Construct input_ids\n        if not batch.forward_mode.is_idle():\n'''
construct_replacement = '''        # Construct input_ids\n        if not batch.forward_mode.is_idle():\n            _mtp_pp_ids = getattr(batch, "_mtp_pp_input_ids", None)\n            if _mtp_pp_ids is not None:\n                if int(_mtp_pp_ids.numel()) != int(batch.input_ids.numel()):\n                    raise RuntimeError(\n                        "[MTP-PP-TRANSPORT-LEN] "\n                        f"transported={int(_mtp_pp_ids.numel())} "\n                        f"local={int(batch.input_ids.numel())}"\n                    )\n                _raw_min = int(batch.input_ids.min().item()) if batch.input_ids.numel() else 0\n                _raw_max = int(batch.input_ids.max().item()) if batch.input_ids.numel() else -1\n                batch.input_ids = _mtp_pp_ids\n                _new_min = int(batch.input_ids.min().item()) if batch.input_ids.numel() else 0\n                _new_max = int(batch.input_ids.max().item()) if batch.input_ids.numel() else -1\n                logger.info(\n                    "[MTP-PP-TRANSPORT-IDS] raw=[%d,%d] transported=[%d,%d] tokens=%d",\n                    _raw_min, _raw_max, _new_min, _new_max, int(batch.input_ids.numel()),\n                )\n\n'''
if "[MTP-PP-TRANSPORT-IDS]" not in s:
    if construct_needle not in s:
        raise RuntimeError("native MTP draft input construction point not found")
    s = s.replace(construct_needle, construct_replacement, 1)

# Guard every explicit draft-prefill block. Base images have had more than one
# symmetric prefill path, so patch all occurrences rather than only the first.
guard_marker = "[MTP-PP-NON-DRAFT-RANK]"
draft_prefill = '''            # Draft prefill\n            with (\n                self.draft_worker.draft_tp_context(\n                    self.draft_worker.draft_runner.tp_group\n                ),\n'''
guarded_prefill = '''            # Draft prefill\n            if self._draft_worker is None:\n                logger.debug("[MTP-PP-NON-DRAFT-RANK] target-only PP stage")\n                return batch_output\n            with (\n                self.draft_worker.draft_tp_context(\n                    self.draft_worker.draft_runner.tp_group\n                ),\n'''
if guard_marker not in s:
    count = s.count(draft_prefill)
    if count == 0:
        raise RuntimeError("native MTP draft-prefill blocks not found")
    s = s.replace(draft_prefill, guarded_prefill)

EAGLE.write_text(s)

# ---------------------------------------------------------------------------
# 3) Preserve a self-contained draft embedding/head on non-owning PP stages.
# ---------------------------------------------------------------------------
s = MTP.read_text()
old = '''        if embed is not None:\n            del self.model.embed_tokens.weight\n            self.model.embed_tokens.weight = embed\n        if head is not None and not self.config.tie_word_embeddings:\n            del self.lm_head.weight\n            self.lm_head.weight = head\n'''
new = '''        def _is_real_endpoint(weight):\n            return (\n                isinstance(weight, torch.Tensor)\n                and weight.numel() > 0\n                and weight.ndim >= 2\n            )\n\n        if _is_real_endpoint(embed):\n            del self.model.embed_tokens.weight\n            self.model.embed_tokens.weight = embed\n        else:\n            logger.info(\n                "[MTP-PP-ENDPOINT] keeping draft embedding; target PP stage "\n                "does not own a real embedding weight"\n            )\n        if _is_real_endpoint(head) and not self.config.tie_word_embeddings:\n            del self.lm_head.weight\n            self.lm_head.weight = head\n        elif not self.config.tie_word_embeddings:\n            logger.info(\n                "[MTP-PP-ENDPOINT] keeping draft lm_head; target PP stage "\n                "does not own a real lm_head weight"\n            )\n'''
if "[MTP-PP-ENDPOINT] keeping draft embedding" not in s:
    if old not in s:
        raise RuntimeError("Qwen3.5 MTP endpoint sharing block not found")
    s = s.replace(old, new, 1)

# Final preflight before embedding lookup. Keep this until PP-native MTP is
# functionally stable so malformed ids fail synchronously instead of poisoning
# the CUDA context.
old_embed = '''            if input_embeds is None:\n                input_embeds = self.model.embed_tokens(input_ids)\n\n            hidden_states = forward_batch.spec_info.hidden_states\n'''
new_embed = '''            if input_embeds is None:\n                _embed_weight = self.model.embed_tokens.weight\n                _rows = int(_embed_weight.shape[0])\n                if input_ids.numel() > 0:\n                    _min_id = int(input_ids.min().item())\n                    _max_id = int(input_ids.max().item())\n                else:\n                    _min_id = 0\n                    _max_id = -1\n                if _min_id < 0 or _max_id >= _rows:\n                    raise RuntimeError(\n                        "[MTP-PP-INPUT-OOB] "\n                        f"input_ids=[{_min_id},{_max_id}] "\n                        f"embed_rows={_rows} "\n                        f"config_vocab={self.config.vocab_size} "\n                        f"mode={forward_batch.forward_mode}"\n                    )\n                logger.info(\n                    "[MTP-PP-INPUT-RANGE] input_ids=[%d,%d] "\n                    "embed_rows=%d config_vocab=%d",\n                    _min_id, _max_id, _rows, int(self.config.vocab_size),\n                )\n                input_embeds = self.model.embed_tokens(input_ids)\n\n            hidden_states = forward_batch.spec_info.hidden_states\n'''
if "[MTP-PP-INPUT-OOB]" not in s:
    if old_embed not in s:
        raise RuntimeError("Qwen3.5 MTP token embedding block not found")
    s = s.replace(old_embed, new_embed, 1)

MTP.write_text(s)
print("PATCHED native MTP PP token transport, draft ownership, and endpoints")
print(SCHED)
print(EAGLE)
print(MTP)
