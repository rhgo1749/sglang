from pathlib import Path

EAGLE = Path("/sgl-workspace/sglang/python/sglang/srt/speculative/eagle_worker_v2.py")
MTP = Path("/sgl-workspace/sglang/python/sglang/srt/models/qwen3_5_mtp.py")
SCHED = Path("/sgl-workspace/sglang/python/sglang/srt/managers/scheduler_pp_mixin.py")
KEY = "__mtp_pp_input_ids"

# ---------------------------------------------------------------------------
# 1) Carry the real token ids from PP0 through the PP proxy tensor transport.
# ScheduleBatch.prepare_for_extend intentionally leaves input_ids=None and keeps
# the real extend ids in prefill_input_ids_cpu until forward-stream H2D.  Snapshot
# that authoritative pinned staging tensor on PP0, not ScheduleBatch.input_ids.
# ---------------------------------------------------------------------------
s = SCHED.read_text()

launch_needle = '''                result = self.run_batch(cur_batch, pp_proxy_tensors)\n'''
launch_replacement = '''                # Native PP-MTP token side-channel. For prefill, SGLang keeps\n                # authoritative token ids in prefill_input_ids_cpu and leaves\n                # ScheduleBatch.input_ids=None until forward-stream resolution.\n                if self.pp_group.is_first_rank:\n                    if cur_batch.forward_mode.is_extend() or cur_batch.is_extend_in_batch:\n                        _mtp_ids = getattr(cur_batch, "prefill_input_ids_cpu", None)\n                        if _mtp_ids is None:\n                            _mtp_ids = getattr(cur_batch, "input_ids", None)\n                        if _mtp_ids is None:\n                            raise RuntimeError(\n                                "[MTP-PP-SOURCE-MISSING] PP0 has neither "\n                                "prefill_input_ids_cpu nor input_ids"\n                            )\n                        cur_batch._mtp_pp_input_ids = _mtp_ids.to(\n                            self.device, non_blocking=True\n                        ).detach().clone()\n                        if cur_batch._mtp_pp_input_ids.numel() > 0:\n                            logger.info(\n                                "[MTP-PP-SOURCE-IDS] ids=[%d,%d] tokens=%d",\n                                int(cur_batch._mtp_pp_input_ids.min().item()),\n                                int(cur_batch._mtp_pp_input_ids.max().item()),\n                                int(cur_batch._mtp_pp_input_ids.numel()),\n                            )\n                elif pp_proxy_tensors is not None:\n                    _mtp_ids = pp_proxy_tensors.tensors.pop("__mtp_pp_input_ids", None)\n                    if _mtp_ids is not None:\n                        cur_batch._mtp_pp_input_ids = _mtp_ids\n\n                result = self.run_batch(cur_batch, pp_proxy_tensors)\n'''
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

# Prefill-side PP constructs the native MTP draft only on the last target
# stage.  Guard at the semantic boundary (immediately after target prefill is
# published), not by matching the wording/format of the following draft block.
# This protects every draft-prefill context below it, including base-image
# variants whose comments or context-manager layout differ.
owner_marker = "[MTP-PP-PREFILL-OWNER]"
fn_start = s.find("    def forward_batch_generation(")
if fn_start < 0:
    raise RuntimeError("forward_batch_generation not found in eagle_worker_v2.py")
fn_end = s.find("\n    def ", fn_start + len("    def forward_batch_generation("))
if fn_end < 0:
    fn_end = len(s)
fn = s[fn_start:fn_end]

if owner_marker not in fn:
    publish_needle = '''            if on_publish is not None:\n                on_publish(batch_output.new_seq_lens)\n'''
    publish_at = fn.find(publish_needle)
    if publish_at < 0:
        raise RuntimeError(
            "native MTP prefill publish boundary not found in forward_batch_generation"
        )
    publish_end = publish_at + len(publish_needle)
    owner_guard = '''\n            # Native PP-MTP ownership boundary: PP0/PP1 are target-only relay\n            # stages. Only the last PP stage owns the colocated draft worker.\n            if self._draft_worker is None:\n                logger.debug("[MTP-PP-PREFILL-OWNER] target-only PP stage")\n                return batch_output\n'''
    fn = fn[:publish_end] + owner_guard + fn[publish_end:]
    s = s[:fn_start] + fn + s[fn_end:]

# Build-time audit: the ownership boundary must occur before the first draft
# context in the prefill branch.  Fail the image build instead of discovering
# a missed textual variant at runtime again.
fn_start = s.find("    def forward_batch_generation(")
fn_end = s.find("\n    def ", fn_start + len("    def forward_batch_generation("))
if fn_end < 0:
    fn_end = len(s)
fn = s[fn_start:fn_end]
guard_at = fn.find(owner_marker)
draft_at = fn.find("self.draft_worker.draft_tp_context(")
if guard_at < 0:
    raise RuntimeError("native MTP PP prefill ownership guard was not installed")
if draft_at < 0:
    raise RuntimeError("native MTP PP draft context not found for ownership audit")
if guard_at > draft_at:
    raise RuntimeError(
        "native MTP PP ownership guard appears after the first draft context"
    )

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
print("VERIFIED [MTP-PP-PREFILL-OWNER] before first draft context")
print(SCHED)
print(EAGLE)
print(MTP)
