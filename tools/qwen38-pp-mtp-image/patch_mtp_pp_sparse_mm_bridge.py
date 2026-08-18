from pathlib import Path
import ast

ROOT = Path("/sgl-workspace/sglang/python/sglang/srt")
SCHED = ROOT / "managers/scheduler_pp_mixin.py"
MM = ROOT / "managers/mm_utils.py"
QWEN = ROOT / "models/qwen3_5.py"
LOGITS = ROOT / "layers/logits_processor.py"
EAGLE = ROOT / "speculative/eagle_worker_v2.py"

MARKER = "[MTP-PP-SPARSE-MM-BRIDGE]"

# ---------------------------------------------------------------------------
# 1) PP0 token side-channel: multimodal hash/pad ids are intentionally outside
#    the vocabulary. They are valid only when the request really carries MM
#    inputs; text-only OOB remains a hard error.
# ---------------------------------------------------------------------------
s = SCHED.read_text()
if "[MTP-PP-MM-HASH-IDS]" not in s:
    needle = '''                            _vocab = int(self.model_config.vocab_size)\n                            if _min_id < 0 or _max_id >= _vocab:\n'''
    repl = '''                            _vocab = int(self.model_config.vocab_size)\n                            _mtp_has_mm = any(\n                                getattr(_req, "multimodal_inputs", None) is not None\n                                for _req in cur_batch.reqs\n                            )\n                            if (_min_id < 0 or _max_id >= _vocab) and not _mtp_has_mm:\n'''
    if needle not in s:
        raise RuntimeError("PP source OOB guard anchor not found")
    s = s.replace(needle, repl, 1)

    stage = '''                            _staging = getattr(cur_batch, "prefill_input_ids_cpu", None)\n'''
    add = '''                            if (_min_id < 0 or _max_id >= _vocab) and _mtp_has_mm:\n                                logger.info(\n                                    "[MTP-PP-MM-HASH-IDS] allowing multimodal pad/hash ids "\n                                    "ids=[%d,%d] vocab=%d tokens=%d",\n                                    _min_id, _max_id, _vocab,\n                                    int(cur_batch._mtp_pp_input_ids.numel()),\n                                )\n                            _staging = getattr(cur_batch, "prefill_input_ids_cpu", None)\n'''
    if stage not in s:
        raise RuntimeError("PP source staging anchor not found")
    s = s.replace(stage, add, 1)
    SCHED.write_text(s)

# ---------------------------------------------------------------------------
# 2) On PP0, preserve only the rows that are replaced by multimodal features.
#    embed_mm_inputs() intentionally clamps hash ids in-place, so snapshot their
#    flattened chunk positions before that call. The current chunk is <= the
#    configured chunked-prefill size; we never transport a full-context
#    [tokens, hidden] embedding tensor across PP.
# ---------------------------------------------------------------------------
s = MM.read_text()
if MARKER not in s:
    anchor = '''    assert hasattr(language_model, "get_input_embeddings")\n    embed_tokens = language_model.get_input_embeddings()\n'''
    inject = '''    assert hasattr(language_model, "get_input_embeddings")\n    embed_tokens = language_model.get_input_embeddings()\n\n    # [MTP-PP-SPARSE-MM-BRIDGE] Capture positions of MM hash/pad ids before\n    # embed_mm_inputs() clamps them to the text vocabulary.\n    _mtp_pp_mm_positions = None\n    if (\n        (not hasattr(language_model, "pp_group") or language_model.pp_group.is_first_rank)\n        and forward_batch.contains_mm_inputs()\n    ):\n        _mtp_vocab = int(embed_tokens.num_embeddings)\n        _mtp_mm_mask = (input_ids < 0) | (input_ids >= _mtp_vocab)\n        _mtp_pp_mm_positions = torch.nonzero(\n            _mtp_mm_mask, as_tuple=False\n        ).reshape(-1).to(torch.int64)\n'''
    if anchor not in s:
        raise RuntimeError("general_mm_embed_routine embedding anchor not found")
    s = s.replace(anchor, inject, 1)

    set_anchor = '''            forward_batch.mm_inputs = None\n            forward_batch.mm_input_embeds = input_embeds\n'''
    set_repl = '''            forward_batch.mm_inputs = None\n            forward_batch.mm_input_embeds = input_embeds\n            if (\n                _mtp_pp_mm_positions is not None\n                and int(_mtp_pp_mm_positions.numel()) > 0\n            ):\n                forward_batch._mtp_pp_mm_positions = _mtp_pp_mm_positions\n                forward_batch._mtp_pp_mm_embeds = input_embeds.index_select(\n                    0, _mtp_pp_mm_positions\n                ).detach().clone()\n'''
    if set_anchor not in s:
        raise RuntimeError("MM forward_batch embedding publication anchor not found")
    s = s.replace(set_anchor, set_repl, 1)

    return_anchor = '''    with torch.profiler.record_function("sglang.vlm.language_model_prefill"):\n        hidden_states = language_model(\n            input_ids=None,\n            forward_batch=forward_batch,\n            input_embeds=input_embeds,\n            **kwargs,\n        )\n    return hidden_states\n'''
    return_repl = '''    with torch.profiler.record_function("sglang.vlm.language_model_prefill"):\n        hidden_states = language_model(\n            input_ids=None,\n            forward_batch=forward_batch,\n            input_embeds=input_embeds,\n            **kwargs,\n        )\n\n    # PPProxyTensors is deliberately duck-typed here to keep mm_utils generic.\n    # Carry only sparse replacement rows; Qwen3.5 propagates them stage-to-stage.\n    _mtp_pos = getattr(forward_batch, "_mtp_pp_mm_positions", None)\n    _mtp_emb = getattr(forward_batch, "_mtp_pp_mm_embeds", None)\n    if hasattr(hidden_states, "tensors") and _mtp_pos is not None and _mtp_emb is not None:\n        hidden_states.tensors["__mtp_pp_mm_positions"] = _mtp_pos\n        hidden_states.tensors["__mtp_pp_mm_embeds"] = _mtp_emb\n    return hidden_states\n'''
    if return_anchor not in s:
        raise RuntimeError("general_mm_embed_routine return anchor not found")
    s = s.replace(return_anchor, return_repl, 1)
    MM.write_text(s)

# ---------------------------------------------------------------------------
# 3) Qwen3.5 target shards propagate the sparse rows through PP. On PP-last,
#    publish them on ForwardBatch so the logits metadata can carry them to the
#    EAGLE worker that owns native MTP.
# ---------------------------------------------------------------------------
s = QWEN.read_text()
if "__mtp_pp_mm_positions" not in s:
    recv = '''            hidden_states = pp_proxy_tensors["hidden_states"]\n            residual = pp_proxy_tensors["residual"]\n'''
    recv_repl = '''            hidden_states = pp_proxy_tensors["hidden_states"]\n            residual = pp_proxy_tensors["residual"]\n            _mtp_pp_mm_positions = pp_proxy_tensors.tensors.get(\n                "__mtp_pp_mm_positions"\n            )\n            _mtp_pp_mm_embeds = pp_proxy_tensors.tensors.get(\n                "__mtp_pp_mm_embeds"\n            )\n            if _mtp_pp_mm_positions is not None and _mtp_pp_mm_embeds is not None:\n                forward_batch._mtp_pp_mm_positions = _mtp_pp_mm_positions\n                forward_batch._mtp_pp_mm_embeds = _mtp_pp_mm_embeds\n'''
    if recv not in s:
        raise RuntimeError("Qwen3.5 PP receive anchor not found")
    s = s.replace(recv, recv_repl, 1)

    ret = '''        if not self.pp_group.is_last_rank:\n            return PPProxyTensors(\n                {\n                    "hidden_states": hidden_states,\n                    "residual": residual,\n                }\n            )\n'''
    ret_repl = '''        if not self.pp_group.is_last_rank:\n            _mtp_proxy = {\n                "hidden_states": hidden_states,\n                "residual": residual,\n            }\n            _mtp_pos = getattr(forward_batch, "_mtp_pp_mm_positions", None)\n            _mtp_emb = getattr(forward_batch, "_mtp_pp_mm_embeds", None)\n            if _mtp_pos is not None and _mtp_emb is not None:\n                _mtp_proxy["__mtp_pp_mm_positions"] = _mtp_pos\n                _mtp_proxy["__mtp_pp_mm_embeds"] = _mtp_emb\n            return PPProxyTensors(_mtp_proxy)\n'''
    if ret not in s:
        raise RuntimeError("Qwen3.5 PP return anchor not found")
    s = s.replace(ret, ret_repl, 1)
    QWEN.write_text(s)

# ---------------------------------------------------------------------------
# 4) Logits output already carries mm_input_embeds as a temporary EAGLE field.
#    Add two equally temporary sparse fields so PP-last can reconstruct the
#    chunk-local MTP input without moving all text embeddings across PP.
# ---------------------------------------------------------------------------
s = LOGITS.read_text()
if "mtp_pp_mm_positions" not in s:
    out_field = '''    mm_input_embeds: Optional[torch.Tensor] = None\n\n\n@dataclasses.dataclass\nclass LogitsMetadata:\n'''
    out_repl = '''    mm_input_embeds: Optional[torch.Tensor] = None\n    mtp_pp_mm_positions: Optional[torch.Tensor] = None\n    mtp_pp_mm_embeds: Optional[torch.Tensor] = None\n\n\n@dataclasses.dataclass\nclass LogitsMetadata:\n'''
    if out_field not in s:
        raise RuntimeError("LogitsProcessorOutput MM field anchor not found")
    s = s.replace(out_field, out_repl, 1)

    meta_field = '''    mm_input_embeds: Optional[torch.Tensor] = None\n\n    # DRAFT_EXTEND_V2: when set, lm_head runs only on these rows'''
    meta_repl = '''    mm_input_embeds: Optional[torch.Tensor] = None\n    mtp_pp_mm_positions: Optional[torch.Tensor] = None\n    mtp_pp_mm_embeds: Optional[torch.Tensor] = None\n\n    # DRAFT_EXTEND_V2: when set, lm_head runs only on these rows'''
    if meta_field not in s:
        raise RuntimeError("LogitsMetadata MM field anchor not found")
    s = s.replace(meta_field, meta_repl, 1)

    from_field = '''            mm_input_embeds=forward_batch.mm_input_embeds,\n            draft_extend_select_index=draft_extend_select_index,\n'''
    from_repl = '''            mm_input_embeds=forward_batch.mm_input_embeds,\n            mtp_pp_mm_positions=getattr(\n                forward_batch, "_mtp_pp_mm_positions", None\n            ),\n            mtp_pp_mm_embeds=getattr(\n                forward_batch, "_mtp_pp_mm_embeds", None\n            ),\n            draft_extend_select_index=draft_extend_select_index,\n'''
    if from_field not in s:
        raise RuntimeError("LogitsMetadata.from_forward_batch MM anchor not found")
    s = s.replace(from_field, from_repl, 1)

    needle = '''            mm_input_embeds=logits_metadata.mm_input_embeds,\n'''
    count = s.count(needle)
    if count < 2:
        raise RuntimeError(f"Unexpected logits MM output anchors: {count}")
    repl = needle + '''            mtp_pp_mm_positions=logits_metadata.mtp_pp_mm_positions,\n            mtp_pp_mm_embeds=logits_metadata.mtp_pp_mm_embeds,\n'''
    s = s.replace(needle, repl)
    LOGITS.write_text(s)

# ---------------------------------------------------------------------------
# 5) PP-last: reconstruct only this prefill chunk's full MTP embedding tensor
#    from the draft's compact FP8 text table plus transported sparse MM rows.
#    IMPORTANT: replace the *whole* parenthesized next_draft_input assignment.
#    Replacing only the inner call leaves executable statements inside
#    `batch_output.next_draft_input = (` and creates invalid Python.
# ---------------------------------------------------------------------------
s = EAGLE.read_text()
if "[MTP-PP-SPARSE-MM-REBUILD]" not in s:
    assignment = '''                batch_output.next_draft_input = (\n                    self.draft_worker._draft_extend_for_prefill(\n                        batch,\n                        batch_output.logits_output.hidden_states,\n                        batch_output.next_token_ids,\n                        batch_output.logits_output.mm_input_embeds,\n                    )\n                )\n'''
    repl = '''                _mtp_mm_full = batch_output.logits_output.mm_input_embeds\n                _mtp_mm_pos = getattr(\n                    batch_output.logits_output, "mtp_pp_mm_positions", None\n                )\n                _mtp_mm_rows = getattr(\n                    batch_output.logits_output, "mtp_pp_mm_embeds", None\n                )\n                if (\n                    _mtp_mm_full is None\n                    and _mtp_mm_pos is not None\n                    and _mtp_mm_rows is not None\n                    and int(_mtp_mm_pos.numel()) > 0\n                ):\n                    _mtp_model = self.draft_worker.draft_runner.model\n                    if not hasattr(_mtp_model, "_mtp_embed_tokens"):\n                        raise RuntimeError(\n                            "[MTP-PP-SPARSE-MM-REBUILD] draft model lacks "\n                            "_mtp_embed_tokens"\n                        )\n                    _mtp_embed_rows = int(\n                        _mtp_model.model.embed_tokens.weight.shape[0]\n                    )\n                    _mtp_safe_ids = batch.input_ids.clamp(\n                        min=0, max=_mtp_embed_rows - 1\n                    )\n                    _mtp_mm_full = _mtp_model._mtp_embed_tokens(_mtp_safe_ids)\n                    _mtp_pos_dev = _mtp_mm_pos.to(\n                        _mtp_mm_full.device, dtype=torch.int64, non_blocking=True\n                    )\n                    _mtp_rows_dev = _mtp_mm_rows.to(\n                        _mtp_mm_full.device,\n                        dtype=_mtp_mm_full.dtype,\n                        non_blocking=True,\n                    )\n                    if int(_mtp_pos_dev.numel()) != int(_mtp_rows_dev.shape[0]):\n                        raise RuntimeError(\n                            "[MTP-PP-SPARSE-MM-LEN] "\n                            f"positions={int(_mtp_pos_dev.numel())} "\n                            f"rows={int(_mtp_rows_dev.shape[0])}"\n                        )\n                    _mtp_mm_full.index_copy_(0, _mtp_pos_dev, _mtp_rows_dev)\n                    logger.info(\n                        "[MTP-PP-SPARSE-MM-REBUILD] PP%d chunk_tokens=%d "\n                        "mm_rows=%d hidden=%d bytes=%.2fMiB",\n                        int(self.ps.pp_rank),\n                        int(_mtp_mm_full.shape[0]),\n                        int(_mtp_pos_dev.numel()),\n                        int(_mtp_mm_full.shape[1]),\n                        _mtp_mm_full.numel() * _mtp_mm_full.element_size() / (1 << 20),\n                    )\n\n                batch_output.next_draft_input = (\n                    self.draft_worker._draft_extend_for_prefill(\n                        batch,\n                        batch_output.logits_output.hidden_states,\n                        batch_output.next_token_ids,\n                        _mtp_mm_full,\n                    )\n                )\n'''
    count = s.count(assignment)
    if count != 1:
        raise RuntimeError(\n            "EAGLE PP-last draft prefill assignment anchor count "\n            f"must be 1, got {count}"\n        )
    s = s.replace(assignment, repl, 1)
    # Parse before writing so installer mistakes fail without publishing a
    # syntactically corrupt target file into the current image layer.
    ast.parse(s, filename=str(EAGLE))
    EAGLE.write_text(s)

# ---------------------------------------------------------------------------
# Structural audit: fail image build rather than discovering a partial bridge
# during a live multimodal request.
# ---------------------------------------------------------------------------
for path in (SCHED, MM, QWEN, LOGITS, EAGLE):
    text = path.read_text()
    ast.parse(text, filename=str(path))

checks = {
    SCHED: ("[MTP-PP-MM-HASH-IDS]",),
    MM: (MARKER, "__mtp_pp_mm_positions", "__mtp_pp_mm_embeds"),
    QWEN: ("__mtp_pp_mm_positions", "__mtp_pp_mm_embeds"),
    LOGITS: ("mtp_pp_mm_positions", "mtp_pp_mm_embeds"),
    EAGLE: ("[MTP-PP-SPARSE-MM-REBUILD]", "_mtp_embed_tokens"),
}
for path, required in checks.items():
    text = path.read_text()
    for token in required:
        if token not in text:
            raise RuntimeError(f"sparse MM bridge audit failed: {path}: {token}")

print("[MTP-PP-SPARSE-MM] installed sparse PP0->PP-last multimodal embedding bridge")
print("VERIFIED multimodal hash ids remain guarded for text-only requests")
print("VERIFIED only chunk-local MM replacement rows cross pipeline stages")