from pathlib import Path
import re

ROOT = Path("/sgl-workspace/sglang/python/sglang/srt")
EAGLE = ROOT / "speculative/eagle_worker_v2.py"
COMMON = ROOT / "speculative/eagle_worker_common.py"
SCHED = ROOT / "managers/scheduler.py"
PP = ROOT / "managers/scheduler_pp_mixin.py"

# This bridge intentionally targets the experiment's topk=1 chain.  The draft
# model remains colocated on the last PP stage.  That stage precomputes the next
# verify proposal; only the tiny token chain is carried backwards through the
# existing PP output ring.  Each target stage rebuilds its own verify mask/KV
# metadata locally, so no target-rank-local mask or hidden tensor crosses ranks.

# ---------------------------------------------------------------------------
# 1) Make shared target verify pipeline-aware: non-last stages run only their
# target shard and return the normal PP hidden-state proxy. Sampling/acceptance
# remains last-rank-only.
# ---------------------------------------------------------------------------
s = COMMON.read_text()
if "[MTP-PP-VERIFY-NONLAST]" not in s:
    sig_old = '''def run_eagle_verify(\n    batch: ScheduleBatch,\n    *,\n    target_worker: TpModelWorker,\n    req_to_token_pool: ReqToTokenPool,\n    token_to_kv_pool_allocator: Any,\n    plan_stream: Any,\n    plan_stream_ctx: Any,\n    topk: int,\n    num_draft_tokens: int,\n    device: str,\n    metadata_ready_pre_pad: bool,\n    finalize_tree_path: bool,\n    grammar_barrier=None,\n) -> GenerationBatchResult:\n'''
    sig_new = sig_old.replace(
        "    grammar_barrier=None,\n",
        "    grammar_barrier=None,\n    pp_proxy_tensors=None,\n",
    )
    if sig_old not in s:
        raise RuntimeError("run_eagle_verify signature not found")
    s = s.replace(sig_old, sig_new, 1)

    call_old = '''    forward_batch_output = target_worker.forward_batch_generation(\n        batch=None,\n        forward_batch=verify_forward_batch,\n        is_verify=True,\n    )\n    logits_output = forward_batch_output.logits_output\n'''
    call_new = '''    forward_batch_output = target_worker.forward_batch_generation(\n        batch=None,\n        forward_batch=verify_forward_batch,\n        pp_proxy_tensors=pp_proxy_tensors,\n        is_verify=True,\n    )\n\n    # PP0/PP1 own target shards only.  They must return the activation proxy\n    # immediately; logits/sampling/acceptance are meaningful only on PP-last.\n    if not target_worker.pp_group.is_last_rank:\n        return GenerationBatchResult(\n            pp_hidden_states_proxy_tensors=(\n                forward_batch_output.pp_hidden_states_proxy_tensors\n            ),\n            can_run_cuda_graph=can_run_cuda_graph,\n            extra_keep_alive_refs=[verify_forward_batch],\n        )\n\n    # [MTP-PP-VERIFY-NONLAST] last rank continues with sampling/acceptance.\n    logits_output = forward_batch_output.logits_output\n'''
    if call_old not in s:
        raise RuntimeError("run_eagle_verify target call not found")
    s = s.replace(call_old, call_new, 1)
COMMON.write_text(s)

# ---------------------------------------------------------------------------
# 2) EAGLEWorkerV2: locate the ACTUAL class method (not the first method with the
# same name elsewhere), rebuild verify metadata locally from a topk=1 token
# bridge, and precompute the next proposal on PP-last after draft-extend.
# ---------------------------------------------------------------------------
s = EAGLE.read_text()
class_at = s.find("class EAGLEWorkerV2(")
if class_at < 0:
    raise RuntimeError("class EAGLEWorkerV2 not found")
fn_at = s.find("    def forward_batch_generation(", class_at)
if fn_at < 0:
    raise RuntimeError("EAGLEWorkerV2.forward_batch_generation not found")

# Ensure ForwardMode is importable for precompute state setup.
if "CaptureHiddenMode, ForwardBatch, ForwardMode" not in s:
    s = s.replace(
        "from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardBatch\n",
        "from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardBatch, ForwardMode\n",
        1,
    )
    if "ForwardMode" not in s[:class_at]:
        # Some base images format the import as a parenthesized block.
        needle = "    CaptureHiddenMode, ForwardBatch\n"
        if needle in s:
            s = s.replace(needle, "    CaptureHiddenMode, ForwardBatch, ForwardMode\n", 1)
if "ForwardMode" not in s[:class_at]:
    raise RuntimeError("could not install ForwardMode import")

# The current base normally imports build_eagle_verify_input already. Fail fast
# rather than silently calling an unavailable helper.
if "build_eagle_verify_input" not in s[:class_at]:
    raise RuntimeError("build_eagle_verify_input import missing from EAGLE worker")

helper_marker = "[MTP-PP-SPEC-BRIDGE]"
if helper_marker not in s[class_at:]:
    helpers = r'''
    def _mtp_pp_is_verify_bridge(self, batch: ScheduleBatch) -> bool:
        spec = getattr(batch, "spec_info", None)
        tokens = getattr(spec, "topk_index", None)
        return (
            self.ps.pp_size > 1
            and batch.forward_mode.is_decode()
            and isinstance(spec, EagleDraftInput)
            and isinstance(tokens, torch.Tensor)
            and tokens.ndim == 2
            and int(tokens.shape[1]) == int(self.speculative_num_draft_tokens)
            and int(self.topk) == 1
        )

    def _mtp_pp_build_verify_from_bridge(self, batch: ScheduleBatch):
        if self.topk != 1:
            raise RuntimeError("[MTP-PP-BRIDGE-TOPK] native PP bridge requires topk=1")
        bridge = batch.spec_info
        verify_tokens = bridge.topk_index
        bs = int(verify_tokens.shape[0])
        expected = int(self.speculative_num_steps + 1)
        if int(verify_tokens.shape[1]) != expected:
            raise RuntimeError(
                "[MTP-PP-BRIDGE-WIDTH] "
                f"tokens={tuple(verify_tokens.shape)} expected_width={expected}"
            )
        # For topk=1 the EAGLE tree is a deterministic chain. The transported
        # tensor is [bonus, draft_1, ..., draft_steps] per request.
        bridge.bonus_tokens = verify_tokens[:, 0].to(torch.int32)
        draft_tokens = verify_tokens[:, 1:].to(torch.int64)
        parent_list = torch.arange(
            -1,
            self.speculative_num_steps - 1,
            dtype=torch.long,
            device=verify_tokens.device,
        ).repeat(bs, 1)
        top_scores_index = torch.arange(
            self.speculative_num_steps,
            dtype=torch.long,
            device=verify_tokens.device,
        ).repeat(bs, 1)
        verify_input = build_eagle_verify_input(
            batch,
            bridge,
            parent_list,
            top_scores_index,
            draft_tokens,
            None,
            target_worker=self.target_worker,
            topk=self.topk,
            num_steps=self.speculative_num_steps,
            num_draft_tokens=self.speculative_num_draft_tokens,
            tree_mask_mode=default_tree_mask_mode(),
            device=self.device,
        )
        logger.info(
            "[MTP-PP-BRIDGE-BUILD] PP%d bs=%d width=%d seq=%s",
            int(self.ps.pp_rank),
            bs,
            int(verify_tokens.shape[1]),
            tuple(batch.seq_lens.shape),
        )
        return verify_input

    def _mtp_pp_precompute_bridge(
        self,
        batch: ScheduleBatch,
        next_draft_input: EagleDraftInput,
        next_seq_lens: torch.Tensor,
        *,
        reserve_after_prefill: bool,
    ) -> EagleDraftInput:
        if self.draft_worker is None:
            raise RuntimeError("[MTP-PP-BRIDGE-OWNER] PP-last draft worker is missing")
        if self.topk != 1:
            raise RuntimeError("[MTP-PP-BRIDGE-TOPK] native PP bridge requires topk=1")
        if self.speculative_num_draft_tokens != self.speculative_num_steps + 1:
            raise RuntimeError(
                "[MTP-PP-BRIDGE-SHAPE] topk=1 requires draft_tokens=steps+1"
            )

        batch.spec_info = next_draft_input
        batch.seq_lens = next_seq_lens
        if batch.seq_lens_cpu is not None:
            batch.seq_lens_cpu = next_seq_lens.to("cpu")
            batch.seq_lens_sum = int(batch.seq_lens_cpu.sum())
        batch.forward_mode = ForwardMode.DECODE
        batch.input_ids = None

        if reserve_after_prefill:
            # The normal decode scheduler has not run eagle_prepare_for_decode
            # yet. Reserve the next target/draft window now so the last-stage
            # draft can be moved one phase earlier. Keep the allocation, but undo
            # the scheduler-visible decode counter increment; the real next
            # iteration will prepare normally on all PP stages.
            from sglang.srt.speculative.eagle_utils import eagle_prepare_for_decode

            decode_batch_idx = [int(r.decode_batch_idx) for r in batch.reqs]
            eagle_prepare_for_decode(batch)
            for req, old_idx in zip(batch.reqs, decode_batch_idx):
                req.decode_batch_idx = old_idx

        with (
            self.draft_worker.draft_tp_context(
                self.draft_worker.draft_runner.tp_group
            ),
            speculative_moe_backend_context(),
            speculative_moe_a2a_backend_context(),
            spec_stage_span("mtp_pp_predraft"),
        ):
            verify_input = self.draft_worker.draft(batch)

        bs = len(batch.reqs)
        verify_tokens = verify_input.draft_token.reshape(
            bs, self.speculative_num_draft_tokens
        ).detach().clone()
        # Reuse EagleDraftInput as a scheduler-safe carrier.  Its normal
        # topk_index width is 1 here; width==draft_tokens (4) is the explicit
        # PP bridge marker and survives filter_batch/merge_batch.
        bridge = EagleDraftInput(
            topk_p=torch.zeros(
                (bs, self.topk), dtype=torch.float32, device=verify_tokens.device
            ),
            topk_index=verify_tokens,
            hidden_states=None,
            bonus_tokens=verify_tokens[:, 0].to(torch.int32),
            capture_hidden_mode=CaptureHiddenMode.LAST,
        )
        logger.info(
            "[MTP-PP-SPEC-BRIDGE] PP%d packed verify tokens shape=%s reserve_prefill=%s",
            int(self.ps.pp_rank),
            tuple(verify_tokens.shape),
            reserve_after_prefill,
        )
        return bridge

    def _mtp_pp_forward_bridge(
        self,
        batch: ScheduleBatch,
        on_publish=None,
        grammar_barrier=None,
        pp_proxy_tensors=None,
    ):
        verify_input = self._mtp_pp_build_verify_from_bridge(batch)
        batch.spec_info = verify_input
        batch_output = self.verify(
            batch,
            grammar_barrier=grammar_barrier,
            pp_proxy_tensors=pp_proxy_tensors,
        )

        if not self.target_worker.pp_group.is_last_rank:
            logger.info(
                "[MTP-PP-VERIFY-STAGE] PP%d target shard complete; forwarding proxy",
                int(self.ps.pp_rank),
            )
            return batch_output

        # PP-last owns sampling and the only native MTP worker.
        if on_publish is not None:
            on_publish(batch_output.new_seq_lens)
        with (
            self.draft_worker.draft_tp_context(
                self.draft_worker.draft_runner.tp_group
            ),
            speculative_moe_backend_context(),
            speculative_moe_a2a_backend_context(),
            spec_stage_span("draft_extend"),
        ):
            self.draft_worker._draft_extend_for_decode(batch, batch_output)

        batch_output.next_draft_input = self._mtp_pp_precompute_bridge(
            batch,
            batch_output.next_draft_input,
            batch_output.new_seq_lens,
            reserve_after_prefill=False,
        )
        logger.info(
            "[MTP-PP-VERIFY-LAST] PP%d accepted=%s next_bridge=%s",
            int(self.ps.pp_rank),
            tuple(batch_output.accept_lens.shape),
            tuple(batch_output.next_draft_input.topk_index.shape),
        )
        return batch_output

    def mtp_pp_commit_nonlast_verify(
        self, batch: ScheduleBatch, batch_result: GenerationBatchResult
    ) -> None:
        """Commit PP-last's topk=1 acceptance into this target stage's GDN state."""
        if self.target_worker.pp_group.is_last_rank:
            return
        accept_cpu = getattr(batch_result, "accept_lens", None)
        if accept_cpu is None or not batch.forward_mode.is_decode():
            return
        from sglang.srt.speculative.spec_utils import (
            commit_mamba_states_after_verify,
            prepare_mamba_track_for_verify,
        )

        accept_lens = accept_cpu.to(self.device, non_blocking=True)
        bs = int(accept_lens.shape[0])
        width = int(batch_result.speculative_num_draft_tokens)
        # topk=1 => global tree node ids are contiguous per request.
        accept_index = torch.arange(
            bs * width, dtype=torch.int32, device=self.device
        ).reshape(bs, width)
        prepare_mamba_track_for_verify(batch)
        commit_mamba_states_after_verify(
            self.target_worker,
            batch,
            accept_lens,
            accept_index,
            width,
        )
        logger.info(
            "[MTP-PP-MAMBA-COMMIT] PP%d accept=%s width=%d",
            int(self.ps.pp_rank),
            accept_lens.tolist(),
            width,
        )

'''
    s = s[:fn_at] + helpers + s[fn_at:]
    class_at = s.find("class EAGLEWorkerV2(")
    fn_at = s.find("    def forward_batch_generation(", class_at)

# Recompute the class method span after helper insertion.
fn_end = s.find("\n    def ", fn_at + len("    def forward_batch_generation("))
if fn_end < 0:
    fn_end = len(s)
fn = s[fn_at:fn_end]

# Route a received precomputed proposal before the ordinary local-draft path.
if "self._mtp_pp_is_verify_bridge(batch)" not in fn:
    body_anchor = "    ):\n"
    body_at = fn.find(body_anchor)
    if body_at < 0:
        raise RuntimeError("EAGLEWorkerV2.forward_batch_generation signature end missing")
    body_at += len(body_anchor)
    route = '''        if self._mtp_pp_is_verify_bridge(batch):\n            return self._mtp_pp_forward_bridge(\n                batch,\n                on_publish=on_publish,\n                grammar_barrier=grammar_barrier,\n                pp_proxy_tensors=pp_proxy_tensors,\n            )\n\n'''
    fn = fn[:body_at] + route + fn[body_at:]

# Ensure target prefill receives the previous PP shard activation.
prefill_call_old = '''            batch_output = self.target_worker.forward_batch_generation(\n                batch, capture_hidden_mode=target_capture_mode\n            )\n'''
prefill_call_new = '''            batch_output = self.target_worker.forward_batch_generation(\n                batch,\n                pp_proxy_tensors=pp_proxy_tensors,\n                capture_hidden_mode=target_capture_mode,\n            )\n'''
if prefill_call_old in fn:
    fn = fn.replace(prefill_call_old, prefill_call_new, 1)
elif "pp_proxy_tensors=pp_proxy_tensors" not in fn[: fn.find("# Draft prefill") if "# Draft prefill" in fn else len(fn)]:
    # Parenthesized/comma-form variants: insert after the batch argument in the
    # first target-worker prefill call.
    call_at = fn.find("self.target_worker.forward_batch_generation(\n")
    if call_at < 0:
        raise RuntimeError("target prefill call not found")
    arg_at = fn.find("                batch,\n", call_at)
    if arg_at < 0:
        raise RuntimeError("target prefill batch arg not found")
    arg_end = arg_at + len("                batch,\n")
    fn = fn[:arg_end] + "                pp_proxy_tensors=pp_proxy_tensors,\n" + fn[arg_end:]

# Semantic ownership guard in the ACTUAL EAGLEWorkerV2 method.
draft_prefill_at = fn.find("# Draft prefill")
if draft_prefill_at < 0:
    draft_prefill_at = fn.find("self.draft_worker._draft_extend_for_prefill(")
if draft_prefill_at < 0:
    raise RuntimeError("EAGLEWorkerV2 draft prefill boundary not found")
pre_draft = fn[max(0, draft_prefill_at - 500):draft_prefill_at]
if "[MTP-PP-PREFILL-OWNER-V2]" not in pre_draft:
    guard = '''            # [MTP-PP-PREFILL-OWNER-V2] Only PP-last owns native MTP.\n            if self.draft_worker is None:\n                logger.info(\n                    "[MTP-PP-PREFILL-OWNER-V2] PP%d target-only prefill relay",\n                    int(self.ps.pp_rank),\n                )\n                return batch_output\n\n'''
    fn = fn[:draft_prefill_at] + guard + fn[draft_prefill_at:]

# On PP-last, turn prefill's ordinary next EagleDraftInput into the proposal for
# the first verify iteration before returning it to the PP output ring.
prefill_call_at = fn.find("self.draft_worker._draft_extend_for_prefill(")
if prefill_call_at < 0:
    raise RuntimeError("_draft_extend_for_prefill call not found")
return_at = fn.find("                return batch_output\n", prefill_call_at)
if return_at < 0:
    raise RuntimeError("draft prefill return not found")
if "reserve_after_prefill=True" not in fn[prefill_call_at:return_at]:
    inject = '''                if self.ps.pp_size > 1:\n                    batch_output.next_draft_input = self._mtp_pp_precompute_bridge(\n                        batch,\n                        batch_output.next_draft_input,\n                        batch_output.new_seq_lens,\n                        reserve_after_prefill=True,\n                    )\n'''
    fn = fn[:return_at] + inject + fn[return_at:]

s = s[:fn_at] + fn + s[fn_end:]

# Patch EAGLEWorkerV2.verify to pass the incoming target PP activation into the
# shared verify helper.
verify_at = s.find("    def verify(self, batch: ScheduleBatch", class_at)
if verify_at < 0:
    raise RuntimeError("EAGLEWorkerV2.verify not found")
verify_end = s.find("\n    def ", verify_at + 8)
if verify_end < 0:
    verify_end = len(s)
verify_fn = s[verify_at:verify_end]
if "pp_proxy_tensors=None" not in verify_fn.split(":", 1)[0]:
    verify_fn = verify_fn.replace(
        "def verify(self, batch: ScheduleBatch, grammar_barrier=None):",
        "def verify(\n        self, batch: ScheduleBatch, grammar_barrier=None, pp_proxy_tensors=None\n    ):",
        1,
    )
if "pp_proxy_tensors=pp_proxy_tensors" not in verify_fn:
    needle = "            grammar_barrier=grammar_barrier,\n"
    if needle not in verify_fn:
        raise RuntimeError("run_eagle_verify grammar_barrier arg not found")
    verify_fn = verify_fn.replace(
        needle,
        needle + "            pp_proxy_tensors=pp_proxy_tensors,\n",
        1,
    )
s = s[:verify_at] + verify_fn + s[verify_end:]
EAGLE.write_text(s)

# ---------------------------------------------------------------------------
# 3) Non-overlap scheduler: PP non-last must only return its activation proxy.
# The PP output ring, not run_batch(), installs last-rank speculative state.
# ---------------------------------------------------------------------------
s = SCHED.read_text()
if "[MTP-PP-RUN-BRIDGE]" not in s:
    start = s.find("            elif not batch.spec_algorithm.is_none():\n")
    if start < 0:
        raise RuntimeError("scheduler non-overlap spec branch not found")
    end = s.find("            else:\n                kwargs = (", start)
    if end < 0:
        raise RuntimeError("scheduler non-overlap spec branch end not found")
    old = s[start:end]
    new = '''            elif not batch.spec_algorithm.is_none():\n                # [MTP-PP-RUN-BRIDGE] Non-overlap PP spec: every target stage\n                # executes forward, but only PP-last owns sampled/spec state.\n                resolve_forward_inputs(batch, self.future_map)\n                with self._forward_isolation(batch, overlap=False):\n                    batch_result = self.model_worker.forward_batch_generation(\n                        batch, pp_proxy_tensors=pp_proxy_tensors\n                    )\n\n                if self.pp_group.is_last_rank:\n                    # The isolation restore reverted the worker's in-forward SB\n                    # edits; re-apply the bridge state produced by PP-last.\n                    batch.spec_info = batch_result.next_draft_input\n                    if batch_result.new_seq_lens is not None:\n                        batch.seq_lens = batch_result.new_seq_lens\n                        if batch.seq_lens_cpu is not None:\n                            batch.seq_lens_cpu = batch_result.new_seq_lens.to("cpu")\n                            batch.seq_lens_sum = int(batch.seq_lens_cpu.sum())\n                    batch.input_ids = None\n                    self.update_cache_from_scheduler(batch, batch_result)\n                    batch_result.copy_done = self.device_module.Event()\n                    batch_result.copy_to_cpu(\n                        return_logprob=batch.return_logprob,\n                        return_hidden_states=batch.return_hidden_states,\n                    )\n                else:\n                    # Do NOT touch next_draft_input or copy sampled outputs here:\n                    # this rank produced only a target activation proxy. The PP\n                    # output ring will deliver PP-last's authoritative result.\n                    logger.debug(\n                        "[MTP-PP-RUN-BRIDGE] PP%d target proxy ready",\n                        int(self.ps.pp_rank),\n                    )\n'''
    s = s[:start] + new + s[end:]
SCHED.write_text(s)

# ---------------------------------------------------------------------------
# 4) Extend the existing PP output ring with a tiny topk=1 verify-token bridge,
# accept counts, and next seq lengths. Rebuild a scheduler-safe EagleDraftInput
# carrier on every receiving target stage.
# ---------------------------------------------------------------------------
s = PP.read_text()
if "[MTP-PP-OUTPUT-BRIDGE]" not in s:
    prep_at = s.find("    def _pp_prepare_tensor_dict(")
    prep_end = s.find("\n    def ", prep_at + 8)
    if prep_at < 0 or prep_end < 0:
        raise RuntimeError("_pp_prepare_tensor_dict not found")
    prep = s[prep_at:prep_end]
    anchor = '''        tensor_dict = {\n            "next_token_ids": result.next_token_ids,\n        }\n\n'''
    if anchor not in prep:
        raise RuntimeError("PP output tensor_dict anchor not found")
    add = '''        # [MTP-PP-OUTPUT-BRIDGE] PP-last -> all target stages.\n        if not batch.spec_algorithm.is_none():\n            _bridge = getattr(result, "next_draft_input", None)\n            _bridge_tokens = getattr(_bridge, "topk_index", None)\n            if (\n                isinstance(_bridge_tokens, torch.Tensor)\n                and _bridge_tokens.ndim == 2\n                and _bridge_tokens.shape[1] > 1\n            ):\n                tensor_dict["__mtp_pp_verify_tokens"] = _bridge_tokens\n                tensor_dict["__mtp_pp_spec_num_draft_tokens"] = int(\n                    _bridge_tokens.shape[1]\n                )\n            if result.new_seq_lens is not None:\n                tensor_dict["__mtp_pp_new_seq_lens"] = result.new_seq_lens\n            if result.accept_lens is not None:\n                tensor_dict["__mtp_pp_accept_lens"] = result.accept_lens\n            if result.block_accept_lens is not None:\n                tensor_dict["__mtp_pp_block_accept_lens"] = result.block_accept_lens\n            if result.cap_lens is not None:\n                tensor_dict["__mtp_pp_cap_lens"] = result.cap_lens\n\n'''
    prep = prep.replace(anchor, anchor + add, 1)
    s = s[:prep_at] + prep + s[prep_end:]

    # Replace the small result reconstruction method wholesale; its upstream
    # shape is intentionally simple and this keeps bridge state installation in
    # one auditable place.
    result_at = s.find("    def _pp_prep_batch_result(")
    result_end = s.find("\n    def ", result_at + 8)
    if result_at < 0 or result_end < 0:
        raise RuntimeError("_pp_prep_batch_result not found")
    old_result = s[result_at:result_end]
    new_result = '''    def _pp_prep_batch_result(\n        self: Scheduler,\n        batch: ScheduleBatch,\n        mb_metadata: PPBatchMetadata,\n        pp_outputs: PPProxyTensors,\n    ):\n        from sglang.srt.managers.scheduler import GenerationBatchResult\n        from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode\n        from sglang.srt.speculative.eagle_info import EagleDraftInput\n\n        logits_output = None\n        extend_input_len_per_req = None\n        extend_logprob_start_len_per_req = None\n\n        if batch.return_logprob:\n            (\n                logits_output,\n                extend_input_len_per_req,\n                extend_logprob_start_len_per_req,\n            ) = get_logprob_from_pp_outputs(pp_outputs)\n\n        _next_gpu = pp_outputs["next_token_ids"].to(torch.int64)\n        self.future_map.stash(\n            batch.req_pool_indices, RelayPayload(bonus_tokens=_next_gpu)\n        )\n        batch.input_ids = None\n\n        _tensors = pp_outputs.tensors\n        _bridge_tokens = _tensors.get("__mtp_pp_verify_tokens")\n        _bridge = None\n        _spec_width = _tensors.get("__mtp_pp_spec_num_draft_tokens")\n        if _bridge_tokens is not None:\n            _bridge_tokens = _bridge_tokens.to(torch.int64)\n            _bridge = EagleDraftInput(\n                topk_p=torch.zeros(\n                    (_bridge_tokens.shape[0], 1),\n                    dtype=torch.float32,\n                    device=_bridge_tokens.device,\n                ),\n                topk_index=_bridge_tokens,\n                hidden_states=None,\n                bonus_tokens=_bridge_tokens[:, 0].to(torch.int32),\n                capture_hidden_mode=CaptureHiddenMode.LAST,\n            )\n            logger.info(\n                "[MTP-PP-BRIDGE-RECV] PP%d tokens=%s",\n                int(self.ps.pp_rank),\n                tuple(_bridge_tokens.shape),\n            )\n\n        _accept = _tensors.get("__mtp_pp_accept_lens")\n        _block_accept = _tensors.get("__mtp_pp_block_accept_lens")\n        _cap = _tensors.get("__mtp_pp_cap_lens")\n        # Spec result processing explicitly requires CPU next_token_ids and\n        # accept_lens. The output ring owns these tiny copies.\n        if not batch.spec_algorithm.is_none():\n            _next_result = _next_gpu.to("cpu")\n            if _accept is not None:\n                _accept = _accept.to("cpu")\n            if _block_accept is not None:\n                _block_accept = _block_accept.to("cpu")\n            if _cap is not None:\n                _cap = _cap.to("cpu")\n        else:\n            _next_result = pp_outputs["next_token_ids"]\n\n        return GenerationBatchResult(\n            logits_output=logits_output,\n            pp_hidden_states_proxy_tensors=None,\n            next_token_ids=_next_result,\n            extend_input_len_per_req=extend_input_len_per_req,\n            extend_logprob_start_len_per_req=extend_logprob_start_len_per_req,\n            can_run_cuda_graph=mb_metadata.can_run_cuda_graph,\n            next_draft_input=_bridge,\n            new_seq_lens=_tensors.get("__mtp_pp_new_seq_lens"),\n            accept_lens=_accept,\n            block_accept_lens=_block_accept,\n            cap_lens=_cap,\n            speculative_num_draft_tokens=(\n                int(_spec_width) if _spec_width is not None else None\n            ),\n        )\n'''
    s = s[:result_at] + new_result + s[result_end:]

    proc_at = s.find("    def _pp_process_batch_result(")
    proc_end = s.find("\n    def ", proc_at + 8)
    if proc_at < 0 or proc_end < 0:
        raise RuntimeError("_pp_process_batch_result not found")
    new_proc = '''    def _pp_process_batch_result(\n        self: Scheduler, batch: ScheduleBatch, output_result: GenerationBatchResult\n    ):\n        # Non-last hybrid-GDN target stages ran verify forward but could not\n        # choose the accepted recurrent state until PP-last returned accept_lens.\n        if (\n            not batch.spec_algorithm.is_none()\n            and output_result.accept_lens is not None\n            and not self.pp_group.is_last_rank\n        ):\n            _commit = getattr(\n                self.model_worker, "mtp_pp_commit_nonlast_verify", None\n            )\n            if _commit is None:\n                raise RuntimeError(\n                    "[MTP-PP-MAMBA-COMMIT] spec bridge commit hook missing"\n                )\n            _commit(batch, output_result)\n\n        self.process_batch_result(batch, output_result)\n\n        # Install PP-last's authoritative next-iteration state only AFTER result\n        # processing, which still needs this batch's pre-verify seq_lens.\n        if not batch.spec_algorithm.is_none():\n            if output_result.next_draft_input is not None:\n                batch.spec_info = output_result.next_draft_input\n            if output_result.new_seq_lens is not None:\n                batch.seq_lens = output_result.new_seq_lens.to(\n                    self.device, non_blocking=True\n                )\n                if batch.seq_lens_cpu is not None:\n                    batch.seq_lens_cpu = batch.seq_lens.to("cpu")\n                    batch.seq_lens_sum = int(batch.seq_lens_cpu.sum())\n            batch.input_ids = None\n'''
    s = s[:proc_at] + new_proc + s[proc_end:]
PP.write_text(s)

# ---------------------------------------------------------------------------
# Build-time audits.  These deliberately target the EAGLEWorkerV2 class span,
# fixing the old false-positive audit that inspected the first method with the
# same name elsewhere in the file.
# ---------------------------------------------------------------------------
e = EAGLE.read_text()
class_at = e.find("class EAGLEWorkerV2(")
fn_at = e.find("    def forward_batch_generation(", class_at)
fn_end = e.find("\n    def ", fn_at + 8)
if fn_end < 0:
    fn_end = len(e)
fn = e[fn_at:fn_end]
for marker in (
    "self._mtp_pp_is_verify_bridge(batch)",
    "[MTP-PP-PREFILL-OWNER-V2]",
    "reserve_after_prefill=True",
):
    if marker not in fn:
        raise RuntimeError(f"EAGLEWorkerV2 bridge audit missing: {marker}")
if fn.find("[MTP-PP-PREFILL-OWNER-V2]") > fn.find(
    "self.draft_worker._draft_extend_for_prefill("
):
    raise RuntimeError("PP ownership guard is after draft prefill call")

if "pp_proxy_tensors=pp_proxy_tensors" not in COMMON.read_text():
    raise RuntimeError("shared verify did not receive PP proxy support")
if "[MTP-PP-RUN-BRIDGE]" not in SCHED.read_text():
    raise RuntimeError("scheduler PP spec bridge missing")
if "[MTP-PP-OUTPUT-BRIDGE]" not in PP.read_text():
    raise RuntimeError("PP output bridge missing")

print("PATCHED PP-aware native MTP speculative bridge (topk=1)")
print("VERIFIED EAGLEWorkerV2 class-scoped ownership + precomputed verify bridge")
print(EAGLE)
print(COMMON)
print(SCHED)
print(PP)
