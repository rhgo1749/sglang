#!/usr/bin/env python3
"""Cut Qwen3.8 EAGLE MTP over to one authoritative TP1 draft on CUDA2.

Environment this patch targets (intentionally narrow / fail-fast):
  * target TP=2 on CUDA0/CUDA1 (RTX 5070 Ti pair)
  * one Qwen3.8 MTP draft on CUDA2 (RTX 5060 Ti)
  * EAGLE topk=1, num_steps=3, num_draft_tokens=4
  * max_running_requests=1
  * radix cache disabled for the first authoritative cutover

The prior WIP proved the CUDA2 model, TP/attention topology, private RoPE,
FlashInfer resource isolation, prefill numerics, and a 3-step proposal exactly
match the colocated TP2 draft.  This patch removes the colocated draft model
from both target ranks and wires the CUDA2 worker into the real prefill -> draft
-> target verify -> draft-extend loop.

Important design points:
  * TP0 alone owns CUDA2. TP1 never constructs a draft model.
  * The sidecar owns a persistent cloned Req, request row, KV cache and Mamba
    state. Target Req objects are never passed to the sidecar pool.
  * Cross-GPU transfers stage through CPU. Draft outputs are copied CUDA2->CPU
    ->CUDA0 and then broadcast over the existing target TP group to CUDA1.
    A CUDA2 tensor is never passed directly to the target NCCL group.
  * Target publish is deliberately delayed until the sidecar has consumed the
    target hidden states. Correctness first; overlap can be recovered later.
  * The sidecar reuses the target's resolved MemoryPoolConfig only for sizing,
    but allocates entirely independent request/KV/Mamba pools on CUDA2.
  * Sidecar CUDA graphs stay disabled for this cutover gate. Target verify
    graphs remain untouched.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

BRANCH = "wip/qwen38-mtp-sidecar-cuda2"
MIN_TOKENS = 65536


def run(*args: str, cwd: Path | None = None) -> None:
    subprocess.run(args, cwd=cwd, check=True)


def replace_once(s: str, old: str, new: str, label: str) -> str:
    if new in s:
        return s
    if old not in s:
        raise RuntimeError(f"cutover patch point not found: {label}")
    return s.replace(old, new, 1)


def patch_eagle(path: Path) -> None:
    s = path.read_text()
    if "def _mtp_cutover_forward(" in s:
        print("authoritative cutover already installed")
        return

    # ---- EagleDraftWorker: explicit authoritative-sidecar construction mode ----
    s = replace_once(
        s,
        '''        target_worker: TpModelWorker,\n    ):\n        super().__init__()\n''',
        '''        target_worker: TpModelWorker,\n        mtp_authoritative_sidecar: bool = False,\n    ):\n        super().__init__()\n''',
        "EagleDraftWorker signature",
    )

    s = replace_once(
        s,
        '''        self.target_worker = target_worker\n\n        # Args for easy access\n        self.device = server_args.device\n''',
        '''        self.target_worker = target_worker\n        self._mtp_authoritative_sidecar = mtp_authoritative_sidecar\n        self._mtp_target_gpu_id = target_worker.ps.gpu_id\n        self._mtp_side_req = None\n        self._mtp_side_rid = None\n        self._mtp_side_seq_len = 0\n        self._mtp_side_draft_input = None\n        self._mtp_side_last_draft_slots = None\n        self._mtp_private_rope_ready = False\n\n        # Args for easy access.  The normal worker historically keeps the generic\n        # string ``cuda`` because each TP process already selected its local GPU.\n        # A sidecar lives in the *same process* as target TP0, so make its device\n        # explicit or later helper allocations silently fall back to CUDA0.\n        self.device = (\n            f"cuda:{gpu_id}"\n            if mtp_authoritative_sidecar and server_args.device == "cuda"\n            else server_args.device\n        )\n''',
        "EagleDraftWorker fields/device",
    )

    # ---- independent sidecar pool, sized to the target's resolved token count ----
    s = replace_once(
        s,
        '''        """Allocate draft KV cache pools (called by scheduler)."""\n        self.req_to_token_pool = req_to_token_pool\n''',
        f'''        """Allocate draft KV cache pools (called by scheduler)."""\n        if self._mtp_authoritative_sidecar:\n            from sglang.srt.distributed.parallel_state import get_self_pp_group\n\n            if memory_pool_config is None:\n                raise RuntimeError(\n                    "MTP cutover requires the target's resolved MemoryPoolConfig"\n                )\n\n            target_gpu_id = self._mtp_target_gpu_id\n            try:\n                torch.cuda.set_device(self.gpu_id)\n                with (\n                    _mtp_sidecar_parallel_context(get_self_pp_group()),\n                    draft_pp_context(),\n                    speculative_moe_backend_context(),\n                    speculative_moe_a2a_backend_context(),\n                ):\n                    # Qwen TP1 sidecar is self-contained (full embedding + native\n                    # quantized lm_head), so NEVER share target CUDA0 parameters.\n                    self.init_token_map()\n                    self.draft_worker.alloc_memory_pool(\n                        memory_pool_config=memory_pool_config\n                    )\n\n                mr = self.draft_runner\n                self.req_to_token_pool = mr.req_to_token_pool\n                self.token_to_kv_pool_allocator = mr.token_to_kv_pool_allocator\n                if int(mr.max_total_num_tokens) < {MIN_TOKENS}:\n                    raise RuntimeError(\n                        f"CUDA2 MTP pool too small for cutover: "\n                        f"{{mr.max_total_num_tokens}} < {MIN_TOKENS}"\n                    )\n                free_b, total_b = torch.cuda.mem_get_info(self.gpu_id)\n                logger.info(\n                    "[MTP-CUTOVER-POOL] CUDA%d side_tokens=%d free=%.2f/%.2f GiB",\n                    self.gpu_id,\n                    mr.max_total_num_tokens,\n                    free_b / (1 << 30),\n                    total_b / (1 << 30),\n                )\n            finally:\n                torch.cuda.set_device(target_gpu_id)\n            return\n\n        self.req_to_token_pool = req_to_token_pool\n''',
        "authoritative pool branch",
    )

    # ---- attention backends under coherent TP1 + private process resources ----
    s = replace_once(
        s,
        '''    def init_attention_backends(self):\n        with (\n            self.draft_tp_context(self.draft_runner.tp_group),\n''',
        '''    def init_attention_backends(self):\n        if self._mtp_authoritative_sidecar:\n            from sglang.srt.distributed.parallel_state import get_self_pp_group\n\n            target_gpu_id = self._mtp_target_gpu_id\n            try:\n                torch.cuda.set_device(self.gpu_id)\n                with (\n                    _mtp_sidecar_parallel_context(get_self_pp_group()),\n                    draft_pp_context(),\n                    speculative_moe_backend_context(),\n                    speculative_moe_a2a_backend_context(),\n                ):\n                    self.draft_worker.init_attention_backends()\n                    self.init_attention_backend()\n                logger.info(\n                    "[MTP-CUTOVER-ATTN] CUDA%d decode=%s extend=%s",\n                    self.gpu_id,\n                    type(self.draft_attn_backend).__name__,\n                    type(self.draft_extend_attn_backend).__name__,\n                )\n            finally:\n                torch.cuda.set_device(target_gpu_id)\n            return\n\n        with (\n            self.draft_tp_context(self.draft_runner.tp_group),\n''',
        "authoritative attention branch",
    )

    # ---- eager-only sidecar initialization; target verify graphs stay normal ----
    s = replace_once(
        s,
        '''    def init_cuda_graphs(self):\n        with (\n            self.draft_tp_context(self.draft_runner.tp_group),\n''',
        '''    def init_cuda_graphs(self):\n        if self._mtp_authoritative_sidecar:\n            from sglang.srt.distributed.parallel_state import get_self_pp_group\n\n            target_gpu_id = self._mtp_target_gpu_id\n            try:\n                torch.cuda.set_device(self.gpu_id)\n                self._mtp_authoritative_ensure_rope()\n                with (\n                    _mtp_sidecar_parallel_context(get_self_pp_group()),\n                    draft_pp_context(),\n                    speculative_moe_backend_context(),\n                    speculative_moe_a2a_backend_context(),\n                ):\n                    # This creates EagerRunner and all runner attributes without\n                    # capturing a normal decode graph for the draft TpModelWorker.\n                    self.draft_worker.init_cuda_graphs(\n                        capture_decode_cuda_graph=False\n                    )\n                self.draft_runner.prefill_cuda_graph_runner = None\n                self.draft_runner.decode_cuda_graph_runner = None\n                self.cuda_graph_runner = None\n                self.cuda_graph_runner_for_draft_extend = None\n                logger.info(\n                    "[MTP-CUTOVER-GRAPH] CUDA%d sidecar=eager-only", self.gpu_id\n                )\n            finally:\n                torch.cuda.set_device(target_gpu_id)\n            return\n\n        with (\n            self.draft_tp_context(self.draft_runner.tp_group),\n''',
        "authoritative graph branch",
    )

    # ---- authoritative sidecar runtime helpers ----
    draft_marker = "    def draft(self, batch: ScheduleBatch):\n"
    if draft_marker not in s:
        raise RuntimeError("draft() insertion point not found")

    helpers = r'''    def _mtp_authoritative_to_side(self, value):
        if value is None or not isinstance(value, torch.Tensor):
            return value
        if value.device.type == "cuda":
            value = value.detach().to("cpu")
        return value.to(self.device)

    def _mtp_authoritative_ensure_rope(self):
        if not self._mtp_authoritative_sidecar or self._mtp_private_rope_ready:
            return
        import copy as _copy

        side_device = torch.device(self.device)
        clones = {}
        users = 0
        for _name, mod in self.draft_runner.model.named_modules():
            rope = getattr(mod, "rotary_emb", None)
            if rope is None:
                continue
            private = clones.get(id(rope))
            if private is None:
                private = _copy.deepcopy(rope).to(side_device)
                cache = getattr(private, "cos_sin_cache", None)
                if isinstance(cache, torch.Tensor):
                    private.cos_sin_cache = cache.to(side_device)
                clones[id(rope)] = private
            mod.rotary_emb = private
            users += 1
        self._mtp_private_rope_ready = True
        logger.info(
            "[MTP-CUTOVER-ROPE] CUDA%d users=%d private=%d",
            self.gpu_id,
            users,
            len(clones),
        )

    def _mtp_authoritative_reset_req(self, target_req):
        import copy as _copy

        # max_running_requests=1: a new rid is a clean epoch for the independent
        # sidecar pools. This also releases any speculative tail slots left by
        # the previous request without involving the target radix tree.
        self.req_to_token_pool.clear()
        self.token_to_kv_pool_allocator.clear()

        req = _copy.copy(target_req)
        req.req_pool_idx = None
        req.mamba_pool_idx = None
        req.mamba_ping_pong_track_buffer = None
        req.mamba_next_track_idx = None
        req.mamba_last_track_idx = None
        req.mamba_last_track_seqlen = None
        req.mamba_branching_seqlen = None
        req.mamba_cow_src_index = None
        req.mamba_needs_clear = False
        req.mamba_lazy_is_insert = True
        req.kv_committed_len = 0
        req.kv = None
        req.extend_batch_idx = 0
        req.decode_batch_idx = 0
        req.prefix_indices = torch.empty(
            (0,), dtype=torch.int64, device=self.device
        )
        req.last_node = None
        req.last_host_node = None
        req.best_match_node = None
        req.cache_protected_len = 0
        req.num_matched_prefix_tokens = 0
        req.host_hit_length = 0
        req.swa_host_hit_length = 0
        req.mamba_host_hit_length = 0
        req.storage_hit_length = 0
        req.skip_lock_node_ids = {}

        rows = self.req_to_token_pool.alloc([req])
        if rows is None:
            raise RuntimeError("CUDA2 MTP request-pool allocation failed")
        self._mtp_side_req = req
        self._mtp_side_rid = target_req.rid
        self._mtp_side_seq_len = 0
        self._mtp_side_draft_input = None
        self._mtp_side_last_draft_slots = None
        logger.info(
            "[MTP-CUTOVER-REQ] CUDA%d rid=%s row=%d reset",
            self.gpu_id,
            target_req.rid,
            int(rows[0]),
        )

    def _mtp_authoritative_get_req(self, target_req):
        if self._mtp_side_req is None or self._mtp_side_rid != target_req.rid:
            self._mtp_authoritative_reset_req(target_req)
        # A few scheduler counters are harmless to mirror and keep helper
        # assertions aligned across chunked-prefill iterations.
        self._mtp_side_req.inflight_middle_chunks = target_req.inflight_middle_chunks
        self._mtp_side_req.extend_batch_idx = target_req.extend_batch_idx
        self._mtp_side_req.decode_batch_idx = target_req.decode_batch_idx
        return self._mtp_side_req

    def _mtp_authoritative_make_batch(self, target_batch, seq_len: int):
        import copy as _copy
        from sglang.srt.managers.schedule_batch import set_mamba_track_indices_from_reqs

        side_req = self._mtp_side_req
        if side_req is None:
            raise RuntimeError("CUDA2 MTP side request is not initialized")

        side_batch = _copy.copy(target_batch)
        side_batch.reqs = [side_req]
        side_batch.req_to_token_pool = self.req_to_token_pool
        side_batch.token_to_kv_pool_allocator = self.token_to_kv_pool_allocator
        side_batch.device = self.device
        side_batch.req_pool_indices_cpu = torch.tensor(
            [side_req.req_pool_idx], dtype=torch.int64
        )
        side_batch.req_pool_indices = side_batch.req_pool_indices_cpu.to(self.device)
        side_batch.seq_lens_cpu = torch.tensor([seq_len], dtype=torch.int64)
        side_batch.seq_lens = side_batch.seq_lens_cpu.to(self.device)
        side_batch.seq_lens_sum = seq_len
        side_batch.orig_seq_lens = self._mtp_authoritative_to_side(
            getattr(target_batch, "orig_seq_lens", None)
        )
        side_batch.mamba_track_mask = self._mtp_authoritative_to_side(
            getattr(target_batch, "mamba_track_mask", None)
        )
        side_batch.mamba_track_seqlens = self._mtp_authoritative_to_side(
            getattr(target_batch, "mamba_track_seqlens", None)
        )
        side_batch.mamba_lazy_spec_track_positions_cpu = None
        if hasattr(self.req_to_token_pool, "mamba_pool"):
            set_mamba_track_indices_from_reqs(side_batch)
            side_batch._collect_deferred_mamba_cow_and_clear([side_req])
        return side_batch

    def mtp_authoritative_prefill(
        self,
        target_batch,
        target_hidden_states,
        next_token_ids,
        mm_input_embeds=None,
    ):
        if len(target_batch.reqs) != 1:
            raise RuntimeError(
                f"MTP cutover currently requires batch size 1, got {len(target_batch.reqs)}"
            )
        from sglang.srt.distributed.parallel_state import get_self_pp_group

        target_gpu_id = self._mtp_target_gpu_id
        try:
            torch.cuda.set_device(self.gpu_id)
            self._mtp_authoritative_ensure_rope()
            side_req = self._mtp_authoritative_get_req(target_batch.reqs[0])

            if target_batch.seq_lens_cpu is not None:
                seq_len = int(target_batch.seq_lens_cpu[0])
            else:
                seq_len = int(target_batch.seq_lens[0].detach().cpu())
            extend_len = int(target_batch.extend_lens[0])
            old_len = self._mtp_side_seq_len
            if seq_len - extend_len != old_len:
                raise RuntimeError(
                    "CUDA2 MTP chunk state diverged: "
                    f"old={old_len} seq={seq_len} extend={extend_len}"
                )

            slots = self.token_to_kv_pool_allocator.alloc(extend_len)
            if slots is None:
                raise RuntimeError(
                    f"CUDA2 MTP prefill KV allocation failed: need={extend_len}"
                )
            self.req_to_token_pool.write(
                (side_req.req_pool_idx, slice(old_len, seq_len)), slots
            )

            side_batch = self._mtp_authoritative_make_batch(target_batch, seq_len)
            side_batch.prefix_lens = [old_len]
            side_batch.extend_lens = [extend_len]
            side_batch.extend_num_tokens = extend_len
            side_batch.input_ids = self._mtp_authoritative_to_side(target_batch.input_ids)
            side_batch.out_cache_loc = slots

            with (
                _mtp_sidecar_parallel_context(get_self_pp_group()),
                draft_pp_context(),
                speculative_moe_backend_context(),
                speculative_moe_a2a_backend_context(),
            ):
                next_draft = self._draft_extend_for_prefill(
                    side_batch,
                    self._mtp_authoritative_to_side(target_hidden_states),
                    self._mtp_authoritative_to_side(next_token_ids),
                    self._mtp_authoritative_to_side(mm_input_embeds),
                )
                torch.cuda.synchronize(self.gpu_id)

            side_req.kv_committed_len = seq_len
            self._mtp_side_seq_len = seq_len
            self._mtp_side_draft_input = next_draft
            logger.info(
                "[MTP-CUTOVER-PREFILL] CUDA%d rid=%s old=%d extend=%d seq=%d",
                self.gpu_id,
                side_req.rid,
                old_len,
                extend_len,
                seq_len,
            )
            return next_draft
        finally:
            torch.cuda.set_device(target_gpu_id)

    def mtp_authoritative_draft_tokens(self, target_batch):
        if len(target_batch.reqs) != 1:
            raise RuntimeError("MTP cutover draft currently requires batch size 1")
        if self._mtp_side_draft_input is None:
            raise RuntimeError("CUDA2 MTP draft state is missing after prefill/extend")
        from sglang.srt.distributed.parallel_state import get_self_pp_group
        from sglang.srt.model_executor.forward_batch_info import ForwardMode

        target_gpu_id = self._mtp_target_gpu_id
        try:
            torch.cuda.set_device(self.gpu_id)
            side_req = self._mtp_authoritative_get_req(target_batch.reqs[0])
            seq_len = self._mtp_side_seq_len

            future = self.token_to_kv_pool_allocator.alloc(self.speculative_num_steps)
            if future is None:
                raise RuntimeError(
                    "CUDA2 MTP speculative KV reservation failed: "
                    f"need={self.speculative_num_steps}"
                )
            self.req_to_token_pool.write(
                (
                    side_req.req_pool_idx,
                    slice(seq_len, seq_len + self.speculative_num_steps),
                ),
                future,
            )
            self._mtp_side_last_draft_slots = future

            side_batch = self._mtp_authoritative_make_batch(target_batch, seq_len)
            side_batch.forward_mode = ForwardMode.DECODE
            side_batch.spec_info = self._mtp_side_draft_input
            side_batch.input_ids = self._mtp_side_draft_input.topk_index.reshape(-1)

            with (
                _mtp_sidecar_parallel_context(get_self_pp_group()),
                draft_pp_context(),
                speculative_moe_backend_context(),
                speculative_moe_a2a_backend_context(),
            ):
                forward_batch, _ = prepare_for_draft(
                    self._mtp_side_draft_input,
                    self.req_to_token_pool,
                    side_batch,
                    None,
                    self.draft_runner,
                    self.topk,
                    self.speculative_num_steps,
                )
                if self.speculative_num_steps > 1:
                    self.draft_attn_backend.init_forward_metadata(forward_batch)
                    forward_batch.mark_forward_metadata_ready()
                _, _, draft_tokens, draft_probs = self.draft_forward(forward_batch)
                if draft_probs is not None:
                    raise RuntimeError(
                        "MTP cutover does not support rejection-sampling draft_probs"
                    )
                torch.cuda.synchronize(self.gpu_id)

            logger.info(
                "[MTP-CUTOVER-DRAFT] CUDA%d seq=%d proposal=%s",
                self.gpu_id,
                seq_len,
                draft_tokens.detach().cpu().tolist(),
            )
            return draft_tokens
        finally:
            torch.cuda.set_device(target_gpu_id)

    def mtp_authoritative_extend_after_verify(self, target_batch, target_result):
        if len(target_batch.reqs) != 1:
            raise RuntimeError("MTP cutover extend currently requires batch size 1")
        import copy as _copy
        from sglang.srt.distributed.parallel_state import get_self_pp_group

        target_gpu_id = self._mtp_target_gpu_id
        accept_cpu = int(target_result.accept_lens[0].detach().cpu())
        old_len = self._mtp_side_seq_len
        try:
            torch.cuda.set_device(self.gpu_id)
            side_req = self._mtp_authoritative_get_req(target_batch.reqs[0])

            # Draft-decode's temporary KV is no longer needed after target verify;
            # draft-extend re-materializes the accepted path from target hidden states.
            if self._mtp_side_last_draft_slots is not None:
                self.token_to_kv_pool_allocator.free(
                    self._mtp_side_last_draft_slots
                )
                self._mtp_side_last_draft_slots = None

            width = self.speculative_num_draft_tokens
            extend_slots = self.token_to_kv_pool_allocator.alloc(width)
            if extend_slots is None:
                raise RuntimeError(
                    f"CUDA2 MTP draft-extend KV allocation failed: need={width}"
                )
            self.req_to_token_pool.write(
                (side_req.req_pool_idx, slice(old_len, old_len + width)),
                extend_slots,
            )

            side_batch = self._mtp_authoritative_make_batch(target_batch, old_len)
            side_batch.out_cache_loc = extend_slots

            side_result = _copy.copy(target_result)
            side_logits = _copy.copy(target_result.logits_output)
            side_logits.hidden_states = self._mtp_authoritative_to_side(
                target_result.logits_output.hidden_states
            )
            side_result.logits_output = side_logits
            side_result.next_token_ids = self._mtp_authoritative_to_side(
                target_result.next_token_ids
            )
            side_result.accept_lens = self._mtp_authoritative_to_side(
                target_result.accept_lens
            )
            side_result.next_draft_input = EagleDraftInput(
                bonus_tokens=self._mtp_authoritative_to_side(
                    target_result.next_draft_input.bonus_tokens
                )
            )

            with (
                _mtp_sidecar_parallel_context(get_self_pp_group()),
                draft_pp_context(),
                speculative_moe_backend_context(),
                speculative_moe_a2a_backend_context(),
            ):
                self._draft_extend_for_decode(side_batch, side_result)
                torch.cuda.synchronize(self.gpu_id)

            if accept_cpu < width:
                self.token_to_kv_pool_allocator.free(extend_slots[accept_cpu:])
            new_len = old_len + accept_cpu
            side_req.kv_committed_len = new_len
            self._mtp_side_seq_len = new_len
            self._mtp_side_draft_input = side_result.next_draft_input
            logger.info(
                "[MTP-CUTOVER-EXTEND] CUDA%d old=%d accept=%d new=%d",
                self.gpu_id,
                old_len,
                accept_cpu,
                new_len,
            )
            return self._mtp_side_draft_input
        finally:
            torch.cuda.set_device(target_gpu_id)

'''
    s = s.replace(draft_marker, helpers + draft_marker, 1)

    # ---- Outer worker: replace colocated draft construction with CUDA2-only TP0 ----
    outer_start = s.find(
        "        # The draft runs where the target's last-layer hidden states"
    )
    outer_end = s.find("        # Adaptive speculative", outer_start)
    if outer_start < 0 or outer_end < 0:
        raise RuntimeError("EAGLEWorkerV2 draft-construction block not found")

    outer_block = r'''        # Authoritative local 3-GPU MTP cutover.
        # Target stays TP2 on CUDA0/CUDA1.  Only target TP0 constructs a draft,
        # and that draft is a self-contained TP1 worker on CUDA2.  TP1 is a
        # relay/verify participant and therefore carries no colocated MTP weights.
        self._hosts_draft = get_pp_group().is_last_rank
        self._mtp_sidecar_authoritative = self._hosts_draft

        if self._mtp_sidecar_authoritative:
            if ps.pp_size != 1:
                raise RuntimeError("MTP cutover currently requires PP=1")
            if ps.tp_size != 2:
                raise RuntimeError(
                    f"MTP cutover requires target TP=2, got {ps.tp_size}"
                )
            if self.topk != 1:
                raise RuntimeError(
                    f"MTP cutover currently requires EAGLE topk=1, got {self.topk}"
                )
            if server_args.speculative_adaptive:
                raise RuntimeError("MTP cutover does not yet support adaptive speculative")
            if get_spec().speculative_use_rejection_sampling:
                raise RuntimeError("MTP cutover does not yet support rejection sampling")
            if not server_args.disable_radix_cache:
                raise RuntimeError(
                    "MTP cutover gate requires --disable-radix-cache so CUDA2 can "
                    "mirror every target prefill chunk without a separate radix tree"
                )
            if torch.cuda.device_count() < 3:
                raise RuntimeError(
                    f"MTP cutover requires 3 visible GPUs, got {torch.cuda.device_count()}"
                )

        self._draft_worker = None
        if self._mtp_sidecar_authoritative and ps.tp_rank == 0:
            from sglang.srt.distributed.parallel_state import get_self_pp_group

            sidecar_gpu_id = 2
            target_gpu_id = ps.gpu_id
            sidecar_ps = ParallelState.trivial(gpu_id=sidecar_gpu_id)
            logger.info(
                "[MTP-CUTOVER] constructing authoritative TP1 draft on CUDA%d (%s)",
                sidecar_gpu_id,
                torch.cuda.get_device_name(sidecar_gpu_id),
            )
            try:
                torch.cuda.set_device(sidecar_gpu_id)
                with (
                    _mtp_sidecar_parallel_context(get_self_pp_group()),
                    draft_pp_context(),
                    speculative_moe_backend_context(),
                    speculative_moe_a2a_backend_context(),
                ):
                    self._draft_worker = EagleDraftWorker(
                        server_args,
                        sidecar_gpu_id,
                        sidecar_ps,
                        nccl_port,
                        target_worker,
                        mtp_authoritative_sidecar=True,
                    )
            finally:
                torch.cuda.set_device(target_gpu_id)
        elif self._mtp_sidecar_authoritative:
            logger.info(
                "[MTP-CUTOVER] target TP rank %d is relay-only; no local draft model",
                ps.tp_rank,
            )

'''
    s = s[:outer_start] + outer_block + s[outer_end:]

    # ---- Outer pool gate and rank-uniform WAR/backend properties ----
    prop_marker = '''    @property\n    def war_fastpath_runner(self):\n'''
    if prop_marker not in s:
        raise RuntimeError("war_fastpath_runner marker not found")
    pool_gate = f'''    def alloc_memory_pool(\n        self,\n        memory_pool_config=None,\n        req_to_token_pool=None,\n        token_to_kv_pool_allocator=None,\n    ):\n        super().alloc_memory_pool(\n            memory_pool_config=memory_pool_config,\n            req_to_token_pool=req_to_token_pool,\n            token_to_kv_pool_allocator=token_to_kv_pool_allocator,\n        )\n        if self._mtp_sidecar_authoritative:\n            target_tokens = int(self.target_worker.model_runner.max_total_num_tokens)\n            if target_tokens < {MIN_TOKENS}:\n                raise RuntimeError(\n                    f"MTP cutover target KV pool too small: "\n                    f"{{target_tokens}} < {MIN_TOKENS}"\n                )\n            side_tokens = (\n                int(self._draft_worker.draft_runner.max_total_num_tokens)\n                if self._draft_worker is not None\n                else -1\n            )\n            logger.info(\n                "[MTP-CUTOVER-POOL] target_rank=%d target_tokens=%d side_tokens=%d",\n                self.ps.tp_rank,\n                target_tokens,\n                side_tokens,\n            )\n\n'''
    s = s.replace(prop_marker, pool_gate + prop_marker, 1)

    s = replace_once(
        s,
        '''    @property\n    def war_fastpath_runner(self):\n        # Per the base contract: the step's last shared-buffer-reading phase is\n        # draft_extend, which runs on the draft runner.\n        return self._draft_worker.draft_runner\n''',
        '''    @property\n    def war_fastpath_runner(self):\n        if self._mtp_sidecar_authoritative:\n            # Sidecar work completes synchronously before publish/return, so the\n            # target runner is the last owner of scheduler-shared CUDA buffers.\n            return self._target_worker.model_runner\n        return self._draft_worker.draft_runner\n''',
        "war_fastpath_runner",
    )

    spec_prop_start = s.find("    @property\n    def spec_v2_attn_backends(self) -> tuple:")
    spec_prop_end = s.find("\n    def init_cuda_graphs(self):", spec_prop_start)
    if spec_prop_start < 0 or spec_prop_end < 0:
        raise RuntimeError("spec_v2_attn_backends block not found")
    spec_prop = '''    @property\n    def spec_v2_attn_backends(self) -> tuple:\n        if self._mtp_sidecar_authoritative:\n            # Keep target-side scheduling decisions rank-uniform. CUDA2 batches\n            # build their own CPU seq-len mirrors explicitly.\n            return (self._target_worker.model_runner.attn_backend,)\n        return (\n            self._target_worker.model_runner.attn_backend,\n            self._draft_worker.draft_attn_backend,\n            self._draft_worker.draft_extend_attn_backend\n            or self._draft_worker.draft_runner.attn_backend,\n        )\n'''
    s = s[:spec_prop_start] + spec_prop + s[spec_prop_end:]

    # ---- Outer relay + authoritative execution loop ----
    forward_marker = '''    def forward_batch_generation(\n        self,\n        batch: ScheduleBatch,\n'''
    if forward_marker not in s:
        raise RuntimeError("outer forward insertion marker not found")

    outer_helpers = r'''    def _mtp_cutover_relay_tensor(self, side_tensor, shape, dtype):
        """CUDA2 -> CPU -> target CUDA0, then TP0 -> TP1 target broadcast."""
        target_device = torch.device("cuda", self.gpu_id)
        torch.cuda.set_device(self.gpu_id)
        if self.ps.tp_rank == 0:
            if side_tensor is None:
                raise RuntimeError("TP0 relay source tensor is missing")
            cpu = side_tensor.detach().to("cpu")
            out = cpu.to(device=target_device, dtype=dtype)
            if tuple(out.shape) != tuple(shape):
                raise RuntimeError(
                    f"TP0 relay shape mismatch: {tuple(out.shape)} != {tuple(shape)}"
                )
        else:
            out = torch.empty(shape, dtype=dtype, device=target_device)
        self.target_worker.model_runner.tp_group.broadcast(out, src=0)
        return out

    def _mtp_cutover_relay_draft_input(self, side_input, bs: int):
        cfg = getattr(
            self.target_worker.model_config.hf_config,
            "text_config",
            self.target_worker.model_config.hf_config,
        )
        hidden_size = int(cfg.hidden_size)
        topk_p = self._mtp_cutover_relay_tensor(
            None if side_input is None else side_input.topk_p,
            (bs, self.topk),
            torch.float32,
        )
        topk_index = self._mtp_cutover_relay_tensor(
            None if side_input is None else side_input.topk_index,
            (bs, self.topk),
            torch.int64,
        )
        hidden = self._mtp_cutover_relay_tensor(
            None if side_input is None else side_input.hidden_states,
            (bs, hidden_size),
            torch.bfloat16,
        )
        bonus = self._mtp_cutover_relay_tensor(
            None if side_input is None else side_input.bonus_tokens,
            (bs,),
            torch.int32,
        )
        return EagleDraftInput(
            topk_p=topk_p,
            topk_index=topk_index,
            hidden_states=hidden,
            bonus_tokens=bonus,
            num_tokens_per_req=1,
            num_tokens_for_logprob_per_req=1,
            capture_hidden_mode=CaptureHiddenMode.LAST,
        )

    def _mtp_cutover_relay_proposal(self, side_tokens, bs: int):
        return self._mtp_cutover_relay_tensor(
            side_tokens,
            (bs, self.speculative_num_steps),
            torch.int64,
        )

    def _mtp_cutover_forward(
        self,
        batch: ScheduleBatch,
        on_publish=None,
        grammar_barrier=None,
        pp_proxy_tensors=None,
    ):
        # Idle overlap batches do not carry a real request or sidecar state.
        if batch.forward_mode.is_idle():
            batch.spec_info = EagleVerifyInput.create_idle_input(
                self.topk,
                self.speculative_num_steps,
                self.speculative_num_draft_tokens,
                self.device,
            )
            return self.verify(batch, grammar_barrier=grammar_barrier)

        if batch.forward_mode.is_extend() or batch.is_extend_in_batch:
            target_capture_mode = (
                CaptureHiddenMode.NULL
                if self.speculative_algorithm.is_standalone()
                else CaptureHiddenMode.FULL
            )
            batch_output = self.target_worker.forward_batch_generation(
                batch,
                pp_proxy_tensors=pp_proxy_tensors,
                capture_hidden_mode=target_capture_mode,
            )
            batch_output.new_seq_lens = batch.seq_lens

            side_next = None
            if self.ps.tp_rank == 0:
                side_next = self._draft_worker.mtp_authoritative_prefill(
                    batch,
                    batch_output.logits_output.hidden_states,
                    batch_output.next_token_ids,
                    batch_output.logits_output.mm_input_embeds,
                )
            # Rank1 waits here while TP0 finishes CUDA2; afterwards both ranks
            # hold a shape-valid target-device EagleDraftInput.
            batch_output.next_draft_input = self._mtp_cutover_relay_draft_input(
                side_next, len(batch.reqs)
            )
            if on_publish is not None:
                on_publish(batch_output.new_seq_lens)
            return batch_output

        if batch.spec_info is None:
            raise RuntimeError(
                "MTP cutover decode entered without relayed EagleDraftInput"
            )

        bs = len(batch.reqs)
        side_tokens = None
        if self.ps.tp_rank == 0:
            side_tokens = self._draft_worker.mtp_authoritative_draft_tokens(batch)
        draft_tokens = self._mtp_cutover_relay_proposal(side_tokens, bs)

        # topk=1 has a deterministic chain topology; only tokens need crossing
        # ranks. Build each target rank's verify metadata locally.
        parent_list = torch.arange(
            -1,
            self.speculative_num_steps - 1,
            dtype=torch.long,
            device=draft_tokens.device,
        ).repeat(bs, 1)
        top_scores_index = torch.arange(
            self.speculative_num_steps,
            dtype=torch.long,
            device=draft_tokens.device,
        ).repeat(bs, 1)
        verify_input = build_eagle_verify_input(
            batch,
            batch.spec_info,
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
        batch.spec_info = verify_input
        batch_output = self.verify(batch, grammar_barrier=grammar_barrier)

        side_next = None
        if self.ps.tp_rank == 0:
            side_next = self._draft_worker.mtp_authoritative_extend_after_verify(
                batch, batch_output
            )
        batch_output.next_draft_input = self._mtp_cutover_relay_draft_input(
            side_next, bs
        )
        if on_publish is not None:
            on_publish(batch_output.new_seq_lens)
        return batch_output

'''
    s = s.replace(forward_marker, outer_helpers + forward_marker, 1)

    # Dispatch to cutover before the preserved stock implementation.
    dispatch_old = '''    ):\n        if batch.forward_mode.is_extend() or batch.is_extend_in_batch:\n'''
    dispatch_new = '''    ):\n        if self._mtp_sidecar_authoritative:\n            return self._mtp_cutover_forward(\n                batch,\n                on_publish=on_publish,\n                grammar_barrier=grammar_barrier,\n                pp_proxy_tensors=pp_proxy_tensors,\n            )\n\n        if batch.forward_mode.is_extend() or batch.is_extend_in_batch:\n'''
    # Only replace the occurrence belonging to the outer forward; start search
    # after the newly inserted cutover helper to avoid touching other methods.
    pos = s.find(forward_marker)
    sub = s[pos:]
    if dispatch_old not in sub:
        raise RuntimeError("outer forward dispatch point not found")
    sub = sub.replace(dispatch_old, dispatch_new, 1)
    s = s[:pos] + sub

    path.write_text(s)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--commit", action="store_true")
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[1]
    patch_dir = Path.home() / "projects" / "sglang-patches"
    eagle_src = patch_dir / "eagle_worker_v2.sidecar-pool-probe.py"
    eagle_dst = repo / "python/sglang/srt/speculative/eagle_worker_v2.py"

    if not eagle_src.exists():
        raise SystemExit(f"missing exact-image source: {eagle_src}")

    backup = eagle_src.with_suffix(eagle_src.suffix + ".before-authoritative-cutover")
    if not backup.exists():
        shutil.copy2(eagle_src, backup)

    patch_eagle(eagle_src)
    shutil.copy2(eagle_src, eagle_dst)

    run("python3", "-m", "py_compile", str(eagle_src))
    run("git", "diff", "--check", cwd=repo)
    print("MTP AUTHORITATIVE CUTOVER PATCHED OK")

    if args.commit:
        run("git", "add", str(eagle_dst.relative_to(repo)), cwd=repo)
        status = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=repo)
        if status.returncode != 0:
            run(
                "git",
                "commit",
                "-m",
                "wip: cut Qwen3.8 MTP over to CUDA2 sidecar",
                cwd=repo,
            )
        run("git", "push", "origin", f"HEAD:{BRANCH}", cwd=repo)
        print("MTP AUTHORITATIVE CUTOVER COMMIT/PUSH OK")


if __name__ == "__main__":
    main()
