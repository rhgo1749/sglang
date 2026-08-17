#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pathlib
import re
import subprocess

HOME = pathlib.Path.home()
REPO = HOME / "projects/sglang-fork"
HOST = HOME / "projects/sglang-patches/eagle_worker_v2.sidecar-pool-probe.py"
FORK = REPO / "python/sglang/srt/speculative/eagle_worker_v2.py"

INIT_OLD = '''        self._mtp_side_req = None
        self._mtp_side_rid = None
        self._mtp_side_seq_len = 0
        self._mtp_side_draft_input = None
        self._mtp_side_last_draft_slots = None
'''
INIT_NEW = '''        # Authoritative CUDA2 state is keyed by target request id.  The first
        # cutover prototype kept one scalar request because max_running_requests=1;
        # parallel serving needs independent Req/KV/Mamba lifetimes per request.
        self._mtp_side_req = None  # legacy compatibility; batch path uses dicts below
        self._mtp_side_rid = None
        self._mtp_side_seq_len = 0
        self._mtp_side_draft_input = None
        self._mtp_side_reqs = {}
        self._mtp_side_seq_lens = {}
        self._mtp_side_target_refs = {}
        # Temporary draft KV belongs to the current synchronous batch, not a req.
        self._mtp_side_last_draft_slots = None
'''

RESET_BLOCK = r'''    def _mtp_authoritative_to_side_draft_input(self, draft_input):
        import copy as _copy

        if draft_input is None:
            return None
        side = _copy.copy(draft_input)
        for name in (
            "topk_p",
            "topk_index",
            "draft_probs",
            "hidden_states",
            "dsa_topk_indices",
            "bonus_tokens",
            "kv_indptr",
            "kv_indices",
            "future_indices",
        ):
            value = getattr(side, name, None)
            if isinstance(value, torch.Tensor):
                setattr(side, name, self._mtp_authoritative_to_side(value))
        return side

    def _mtp_authoritative_release_finished(self, keep_rids=()):
        """Reclaim independent CUDA2 Req/KV/Mamba state after target Req finishes."""
        keep = set(keep_rids)
        for rid, req in list(self._mtp_side_reqs.items()):
            if rid in keep:
                continue
            target_ref = self._mtp_side_target_refs.get(rid)
            if target_ref is None or not target_ref.finished():
                continue

            seq_len = int(self._mtp_side_seq_lens.get(rid, 0))
            row = req.req_pool_idx
            if row is not None and seq_len > 0:
                locs = self.req_to_token_pool.req_to_token[row, :seq_len].clone()
                locs = locs[locs > 0]
                if locs.numel() > 0:
                    self.token_to_kv_pool_allocator.free(locs)
            if (
                hasattr(self.req_to_token_pool, "free_mamba_cache")
                and req.mamba_pool_idx is not None
            ):
                self.req_to_token_pool.free_mamba_cache(req)
            if req.req_pool_idx is not None:
                self.req_to_token_pool.free(req)

            self._mtp_side_reqs.pop(rid, None)
            self._mtp_side_seq_lens.pop(rid, None)
            self._mtp_side_target_refs.pop(rid, None)
            logger.info("[MTP-CUTOVER-REQ] CUDA%d rid=%s released", self.gpu_id, rid)

    def _mtp_authoritative_reset_req(self, target_req):
        import copy as _copy

        # Do NOT clear the whole sidecar pool here: other requests may be decoding.
        self._mtp_authoritative_release_finished(keep_rids=(target_req.rid,))

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
        req.prefix_indices = torch.empty((0,), dtype=torch.int64, device=self.device)
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
            # A completed target Req may only have become visible since the first GC.
            self._mtp_authoritative_release_finished(keep_rids=(target_req.rid,))
            rows = self.req_to_token_pool.alloc([req])
        if rows is None:
            raise RuntimeError("CUDA2 MTP request-pool allocation failed")

        self._mtp_side_reqs[target_req.rid] = req
        self._mtp_side_seq_lens[target_req.rid] = 0
        self._mtp_side_target_refs[target_req.rid] = target_req
        logger.info(
            "[MTP-CUTOVER-REQ] CUDA%d rid=%s row=%d allocated active=%d",
            self.gpu_id,
            target_req.rid,
            int(rows[0]),
            len(self._mtp_side_reqs),
        )
        return req
'''

GET_BLOCK = r'''    def _mtp_authoritative_get_req(self, target_req):
        req = self._mtp_side_reqs.get(target_req.rid)
        if req is None:
            req = self._mtp_authoritative_reset_req(target_req)
        self._mtp_side_target_refs[target_req.rid] = target_req
        # Mirror scheduler counters used by chunked-prefill assertions.
        req.inflight_middle_chunks = target_req.inflight_middle_chunks
        req.extend_batch_idx = target_req.extend_batch_idx
        req.decode_batch_idx = target_req.decode_batch_idx
        return req
'''

MAKE_BATCH_BLOCK = r'''    def _mtp_authoritative_make_batch(self, target_batch, seq_lens):
        import copy as _copy
        from sglang.srt.managers.schedule_batch import set_mamba_track_indices_from_reqs

        if len(seq_lens) != len(target_batch.reqs):
            raise RuntimeError(
                f"CUDA2 MTP seq-len batch mismatch: {len(seq_lens)} != {len(target_batch.reqs)}"
            )
        keep_rids = [r.rid for r in target_batch.reqs]
        self._mtp_authoritative_release_finished(keep_rids=keep_rids)
        side_reqs = [self._mtp_authoritative_get_req(r) for r in target_batch.reqs]

        side_batch = _copy.copy(target_batch)
        side_batch.reqs = side_reqs
        side_batch.req_to_token_pool = self.req_to_token_pool
        side_batch.token_to_kv_pool_allocator = self.token_to_kv_pool_allocator
        side_batch.device = self.device
        side_batch.req_pool_indices_cpu = torch.tensor(
            [r.req_pool_idx for r in side_reqs], dtype=torch.int64
        )
        side_batch.req_pool_indices = side_batch.req_pool_indices_cpu.to(self.device)
        side_batch.seq_lens_cpu = torch.tensor(seq_lens, dtype=torch.int64)
        side_batch.seq_lens = side_batch.seq_lens_cpu.to(self.device)
        side_batch.seq_lens_sum = int(sum(seq_lens))
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
            side_batch._collect_deferred_mamba_cow_and_clear(side_reqs)
        return side_batch
'''

PREFILL_BLOCK = r'''    def mtp_authoritative_prefill(
        self,
        target_batch,
        target_hidden_states,
        next_token_ids,
        mm_input_embeds=None,
    ):
        from sglang.srt.distributed.parallel_state import get_self_pp_group

        bs = len(target_batch.reqs)
        if bs == 0:
            raise RuntimeError("MTP cutover prefill received an empty batch")
        target_gpu_id = self._mtp_target_gpu_id
        try:
            torch.cuda.set_device(self.gpu_id)
            self._mtp_authoritative_ensure_rope()
            side_reqs = [self._mtp_authoritative_get_req(r) for r in target_batch.reqs]

            if target_batch.seq_lens_cpu is not None:
                seq_lens = [int(x) for x in target_batch.seq_lens_cpu.tolist()]
            else:
                seq_lens = [int(x) for x in target_batch.seq_lens.detach().cpu().tolist()]
            extend_lens = [int(x) for x in target_batch.extend_lens]
            if len(extend_lens) != bs:
                raise RuntimeError(f"CUDA2 MTP extend_lens mismatch: {len(extend_lens)} != {bs}")
            old_lens = [int(self._mtp_side_seq_lens.get(r.rid, 0)) for r in target_batch.reqs]
            for req, old_len, seq_len, extend_len in zip(
                target_batch.reqs, old_lens, seq_lens, extend_lens
            ):
                if seq_len - extend_len != old_len:
                    raise RuntimeError(
                        "CUDA2 MTP chunk state diverged: "
                        f"rid={req.rid} old={old_len} seq={seq_len} extend={extend_len}"
                    )

            total_extend = int(sum(extend_lens))
            slots = self.token_to_kv_pool_allocator.alloc(total_extend)
            if slots is None:
                raise RuntimeError(
                    f"CUDA2 MTP prefill KV allocation failed: need={total_extend}"
                )
            off = 0
            for side_req, old_len, seq_len, extend_len in zip(
                side_reqs, old_lens, seq_lens, extend_lens
            ):
                if extend_len:
                    self.req_to_token_pool.write(
                        (side_req.req_pool_idx, slice(old_len, seq_len)),
                        slots[off : off + extend_len],
                    )
                off += extend_len

            side_batch = self._mtp_authoritative_make_batch(target_batch, seq_lens)
            side_batch.prefix_lens = old_lens
            side_batch.extend_lens = extend_lens
            side_batch.extend_num_tokens = total_extend
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

            for side_req, target_req, seq_len in zip(
                side_reqs, target_batch.reqs, seq_lens
            ):
                side_req.kv_committed_len = seq_len
                self._mtp_side_seq_lens[target_req.rid] = seq_len
            logger.info(
                "[MTP-CUTOVER-PREFILL] CUDA%d bs=%d seq=%s extend=%s active=%d",
                self.gpu_id,
                bs,
                seq_lens,
                extend_lens,
                len(self._mtp_side_reqs),
            )
            return next_draft
        finally:
            torch.cuda.set_device(target_gpu_id)
'''

DRAFT_BLOCK = r'''    def mtp_authoritative_draft_tokens(self, target_batch):
        from sglang.srt.distributed.parallel_state import get_self_pp_group
        from sglang.srt.model_executor.forward_batch_info import ForwardMode

        bs = len(target_batch.reqs)
        if bs == 0:
            raise RuntimeError("MTP cutover draft received an empty batch")
        if target_batch.spec_info is None:
            raise RuntimeError("CUDA2 MTP draft state is missing after prefill/extend")
        target_gpu_id = self._mtp_target_gpu_id
        try:
            torch.cuda.set_device(self.gpu_id)
            side_reqs = [self._mtp_authoritative_get_req(r) for r in target_batch.reqs]
            seq_lens = [int(self._mtp_side_seq_lens[r.rid]) for r in target_batch.reqs]
            side_input = self._mtp_authoritative_to_side_draft_input(target_batch.spec_info)

            future = self.token_to_kv_pool_allocator.alloc(bs * self.speculative_num_steps)
            if future is None:
                raise RuntimeError(
                    "CUDA2 MTP speculative KV reservation failed: "
                    f"need={bs * self.speculative_num_steps}"
                )
            off = 0
            for side_req, seq_len in zip(side_reqs, seq_lens):
                req_future = future[off : off + self.speculative_num_steps]
                self.req_to_token_pool.write(
                    (
                        side_req.req_pool_idx,
                        slice(seq_len, seq_len + self.speculative_num_steps),
                    ),
                    req_future,
                )
                off += self.speculative_num_steps
            self._mtp_side_last_draft_slots = future

            side_batch = self._mtp_authoritative_make_batch(target_batch, seq_lens)
            side_batch.forward_mode = ForwardMode.DECODE
            side_batch.spec_info = side_input
            side_batch.input_ids = side_input.topk_index.reshape(-1)

            with (
                _mtp_sidecar_parallel_context(get_self_pp_group()),
                draft_pp_context(),
                speculative_moe_backend_context(),
                speculative_moe_a2a_backend_context(),
            ):
                forward_batch, _ = prepare_for_draft(
                    side_input,
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
                "[MTP-CUTOVER-DRAFT] CUDA%d bs=%d seq=%s proposal=%s",
                self.gpu_id,
                bs,
                seq_lens,
                draft_tokens.detach().cpu().tolist(),
            )
            return draft_tokens
        finally:
            torch.cuda.set_device(target_gpu_id)
'''

EXTEND_BLOCK = r'''    def mtp_authoritative_extend_after_verify(self, target_batch, target_result):
        import copy as _copy
        from sglang.srt.distributed.parallel_state import get_self_pp_group

        bs = len(target_batch.reqs)
        if bs == 0:
            raise RuntimeError("MTP cutover extend received an empty batch")
        target_gpu_id = self._mtp_target_gpu_id
        accept_cpu = [int(x) for x in target_result.accept_lens.detach().cpu().tolist()]
        old_lens = [int(self._mtp_side_seq_lens[r.rid]) for r in target_batch.reqs]
        try:
            torch.cuda.set_device(self.gpu_id)
            side_reqs = [self._mtp_authoritative_get_req(r) for r in target_batch.reqs]

            # Draft-decode's temporary KV is no longer needed after target verify.
            if self._mtp_side_last_draft_slots is not None:
                self.token_to_kv_pool_allocator.free(self._mtp_side_last_draft_slots)
                self._mtp_side_last_draft_slots = None

            width = self.speculative_num_draft_tokens
            extend_slots = self.token_to_kv_pool_allocator.alloc(bs * width)
            if extend_slots is None:
                raise RuntimeError(
                    f"CUDA2 MTP draft-extend KV allocation failed: need={bs * width}"
                )
            slot_rows = extend_slots.view(bs, width)
            for side_req, old_len, row_slots in zip(side_reqs, old_lens, slot_rows):
                self.req_to_token_pool.write(
                    (side_req.req_pool_idx, slice(old_len, old_len + width)),
                    row_slots,
                )

            side_batch = self._mtp_authoritative_make_batch(target_batch, old_lens)
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

            free_parts = []
            new_lens = []
            for target_req, side_req, old_len, accepted, row_slots in zip(
                target_batch.reqs, side_reqs, old_lens, accept_cpu, slot_rows
            ):
                if accepted < width:
                    free_parts.append(row_slots[accepted:])
                new_len = old_len + accepted
                new_lens.append(new_len)
                side_req.kv_committed_len = new_len
                self._mtp_side_seq_lens[target_req.rid] = new_len
            if free_parts:
                tail = torch.cat([x for x in free_parts if x.numel() > 0])
                if tail.numel() > 0:
                    self.token_to_kv_pool_allocator.free(tail)

            logger.info(
                "[MTP-CUTOVER-EXTEND] CUDA%d bs=%d old=%s accept=%s new=%s",
                self.gpu_id,
                bs,
                old_lens,
                accept_cpu,
                new_lens,
            )
            return side_result.next_draft_input
        finally:
            torch.cuda.set_device(target_gpu_id)
'''


def replace_method(src: str, name: str, replacement: str) -> str:
    start = src.find(f"    def {name}(")
    if start < 0:
        raise RuntimeError(f"method not found: {name}")
    match = re.search(r"\n    def [A-Za-z_][A-Za-z0-9_]*\(", src[start + 1 :])
    if not match:
        raise RuntimeError(f"next method not found after: {name}")
    end = start + 1 + match.start() + 1
    return src[:start] + replacement.rstrip() + "\n\n" + src[end:]


def patch_source(path: pathlib.Path) -> bool:
    src = path.read_text()
    if "[MTP-CUTOVER-PREFILL] CUDA%d bs=%d seq=%s extend=%s active=%d" in src:
        print(f"parallel3 already fixed: {path}")
        return False

    if "self._mtp_side_reqs = {}" not in src:
        if INIT_OLD not in src:
            raise RuntimeError(f"authoritative init anchor not found: {path}")
        src = src.replace(INIT_OLD, INIT_NEW, 1)

    src = replace_method(src, "_mtp_authoritative_reset_req", RESET_BLOCK)
    src = replace_method(src, "_mtp_authoritative_get_req", GET_BLOCK)
    src = replace_method(src, "_mtp_authoritative_make_batch", MAKE_BATCH_BLOCK)
    src = replace_method(src, "mtp_authoritative_prefill", PREFILL_BLOCK)
    src = replace_method(src, "mtp_authoritative_draft_tokens", DRAFT_BLOCK)
    src = replace_method(src, "mtp_authoritative_extend_after_verify", EXTEND_BLOCK)
    path.write_text(src)
    subprocess.run(["python3", "-m", "py_compile", str(path)], check=True)
    print(f"fixed batch-aware authoritative sidecar: {path}")
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--commit", action="store_true")
    args = ap.parse_args()

    host_changed = patch_source(HOST)
    fork_changed = patch_source(FORK)
    if host_changed and not fork_changed:
        FORK.write_text(HOST.read_text())
        subprocess.run(["python3", "-m", "py_compile", str(FORK)], check=True)
        fork_changed = True
        print(f"synced runtime source: {FORK}")
    elif fork_changed and not host_changed:
        HOST.write_text(FORK.read_text())
        subprocess.run(["python3", "-m", "py_compile", str(HOST)], check=True)
        host_changed = True
        print(f"synced host runtime source: {HOST}")

    print(
        f"MTP CUTOVER PARALLEL3 HOTFIX OK host_changed={host_changed} fork_changed={fork_changed}"
    )

    if args.commit and fork_changed:
        subprocess.run(["git", "-C", str(REPO), "add", str(FORK.relative_to(REPO))], check=True)
        diff = subprocess.run(
            ["git", "-C", str(REPO), "diff", "--cached", "--quiet"],
            check=False,
        )
        if diff.returncode != 0:
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(REPO),
                    "commit",
                    "-m",
                    "feat: make CUDA2 MTP cutover batch-aware",
                ],
                check=True,
            )
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(REPO),
                    "push",
                    "origin",
                    "HEAD:wip/qwen38-mtp-sidecar-cuda2",
                ],
                check=True,
            )
            print("MTP CUTOVER PARALLEL3 COMMIT/PUSH OK")


if __name__ == "__main__":
    main()
