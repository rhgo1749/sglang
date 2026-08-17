#!/usr/bin/env python3
"""Patch the exact-image EAGLE worker with a persistent CUDA2 MTP shadow.

This is the last correctness gate before removing the colocated draft model.
The already-proven sidecar prefill + 3-step proposal is kept authoritative only
for diagnostics; the stock draft still serves requests.  This patch persists the
independent CUDA2 Req/KV/Mamba state across target verify iterations, mirrors
_draft_extend_for_decode on the sidecar, immediately builds the next 3-step
proposal, and compares it with the stock proposal on the next draft iteration.

Scope is intentionally narrow for the current production target:
  * bs == 1
  * topk == 1
  * page_size == 1
  * CPU staging between CUDA0 and CUDA2 (no P2P/IPC assumption)

Rejected speculative tail slots are overwritten rather than reclaimed in this
shadow gate.  That bounded leak is acceptable for the short persistence test;
production cutover will add explicit sidecar slot reclamation.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

BRANCH = "wip/qwen38-mtp-sidecar-cuda2"


def run(*args: str, cwd: Path | None = None) -> None:
    subprocess.run(args, cwd=cwd, check=True)


def patch_eagle(path: Path) -> None:
    s = path.read_text()
    if "[MTP-SIDECAR-PERSIST]" in s:
        print("persistent sidecar shadow already installed")
        return

    # Persist the objects that the one-shot prefill probe already constructed.
    # They own independent CUDA2 request/KV/Mamba state and must survive target
    # verify so we can mirror the stock draft-extend transition.
    state_marker = '''            logger.info(\n                "[MTP-SIDECAR-MULTISTEP] CUDA%d proposal=%s",\n                sidecar_gpu_id,\n                self._mtp_sidecar_multistep_proposal.tolist(),\n            )\n\n            # Save a tiny one-shot correctness snapshot on host memory.  The\n'''
    state_replacement = '''            logger.info(\n                "[MTP-SIDECAR-MULTISTEP] CUDA%d proposal=%s",\n                sidecar_gpu_id,\n                self._mtp_sidecar_multistep_proposal.tolist(),\n            )\n\n            # Persist the CUDA2 request/pool state across target verify.  The\n            # side_batch has already been converted to a DECODE view and the\n            # side_req owns its own HybridReqToTokenPool/Mamba slots.\n            self._mtp_sidecar_persist_state = {\n                "req": side_req,\n                "batch": side_batch,\n                "req_idx": side_req_idx,\n                "seq_len": seq_len,\n                "draft_input": _side_draft_input,\n                "iteration": 0,\n            }\n            logger.info(\n                "[MTP-SIDECAR-PERSIST] armed req_row=%d seq_len=%d",\n                side_req_idx,\n                seq_len,\n            )\n\n            # Save a tiny one-shot correctness snapshot on host memory.  The\n'''
    if state_marker not in s:
        raise RuntimeError("prefill persistence insertion point not found")
    s = s.replace(state_marker, state_replacement, 1)

    # Add two helpers before the stock _draft_extend_for_decode method:
    #  1) build the next sidecar multi-step proposal from persistent state
    #  2) mirror stock draft-extend after target verify, compare the selected
    #     next draft seed, then invoke (1) so the next stock draft() compares
    #     proposal tokens using the comparator already installed.
    method_marker = '''    def _draft_extend_for_decode(\n        self, batch: ScheduleBatch, batch_result: GenerationBatchResult\n    ):\n'''
    helpers = r'''    def _mtp_sidecar_persistent_proposal(self):
        state = getattr(self, "_mtp_sidecar_persist_state", None)
        sidecar = getattr(self, "_mtp_sidecar_probe", None)
        if state is None or sidecar is None:
            return

        if self.topk != 1:
            raise RuntimeError("persistent CUDA2 shadow currently requires topk=1")

        from sglang.srt.distributed.parallel_state import get_self_pp_group
        from sglang.srt.model_executor.forward_batch_info import ForwardMode as _ForwardMode

        sidecar_gpu_id = sidecar.gpu_id
        target_gpu_id = self.ps.gpu_id
        sidecar_device = f"cuda:{sidecar_gpu_id}"
        mr = sidecar.model_runner
        side_req_pool = mr.req_to_token_pool
        side_kv_alloc = mr.token_to_kv_pool_allocator
        side_batch = state["batch"]
        side_req_idx = int(state["req_idx"])
        seq_len = int(state["seq_len"])
        side_input = state["draft_input"]

        try:
            torch.cuda.set_device(sidecar_gpu_id)

            # prepare_for_draft reads future cache locations directly from the
            # request row.  Reserve a fresh TOPK=1 chain after the committed
            # prefix.  During this short shadow test rejected old tail slots may
            # remain allocated; the production cutover will reclaim them.
            future = side_kv_alloc.alloc(int(self.speculative_num_steps))
            if future is None:
                raise RuntimeError(
                    "CUDA2 persistent shadow could not reserve draft KV slots"
                )
            side_req_pool.write(
                (
                    side_req_idx,
                    slice(seq_len, seq_len + int(self.speculative_num_steps)),
                ),
                future,
            )

            side_batch.spec_info = side_input
            side_batch.forward_mode = _ForwardMode.DECODE
            side_batch.input_ids = side_input.topk_index.reshape(-1)
            side_batch.seq_lens_cpu = torch.tensor([seq_len], dtype=torch.int64)
            side_batch.seq_lens = side_batch.seq_lens_cpu.to(sidecar_device)
            side_batch.seq_lens_sum = seq_len

            with (
                _mtp_sidecar_parallel_context(get_self_pp_group()),
                draft_pp_context(),
                speculative_moe_backend_context(),
                speculative_moe_a2a_backend_context(),
            ):
                side_fb, _ = prepare_for_draft(
                    side_input,
                    side_req_pool,
                    side_batch,
                    None,
                    mr,
                    self.topk,
                    self.speculative_num_steps,
                )
                side_fb.return_logprob = False
                side_attn = self._mtp_sidecar_draft_attn_backend
                if self.speculative_num_steps > 1:
                    side_attn.init_forward_metadata(side_fb)
                    side_fb.mark_forward_metadata_ready()

                out_cache = per_step_draft_out_cache_loc(
                    side_fb.out_cache_loc,
                    side_fb.batch_size,
                    self.topk,
                    self.speculative_num_steps,
                )
                cur_token = side_input.topk_index.reshape(-1, 1).to(torch.int64)
                cur_hidden = side_input.hidden_states
                proposal = [cur_token.reshape(-1)]

                for i in range(self.speculative_num_steps - 1):
                    side_fb.input_ids = cur_token.reshape(-1)
                    side_fb.out_cache_loc = out_cache[i]
                    side_fb.spec_info.hidden_states = cur_hidden
                    with forward_context(
                        ForwardContext(attn_backend=side_attn.attn_backends[i])
                    ):
                        step_logits = mr.forward(side_fb).logits_output
                    torch.cuda.synchronize(sidecar_gpu_id)
                    cur_token = torch.argmax(
                        step_logits.next_token_logits, dim=-1, keepdim=True
                    ).to(torch.int64)
                    proposal.append(cur_token.reshape(-1))
                    cur_hidden = step_logits.hidden_states
                    side_fb.positions.add_(1)

            self._mtp_sidecar_multistep_proposal = (
                torch.stack(proposal, dim=1).detach().cpu()
            )
            state["iteration"] = int(state.get("iteration", 0)) + 1
            logger.info(
                "[MTP-SIDECAR-PERSIST] iter=%d seq_len=%d proposal=%s",
                state["iteration"],
                seq_len,
                self._mtp_sidecar_multistep_proposal.tolist(),
            )
        finally:
            torch.cuda.set_device(target_gpu_id)

    def _mtp_sidecar_shadow_extend_for_decode(
        self, batch: ScheduleBatch, batch_result: GenerationBatchResult
    ):
        state = getattr(self, "_mtp_sidecar_persist_state", None)
        sidecar = getattr(self, "_mtp_sidecar_probe", None)
        if state is None or sidecar is None or batch.forward_mode.is_idle():
            return
        if len(batch.reqs) != 1 or self.topk != 1:
            logger.warning(
                "[MTP-SIDECAR-PERSIST] disabling: only bs1/topk1 is supported"
            )
            self._mtp_sidecar_persist_state = None
            return

        import copy as _copy
        from sglang.srt.distributed.parallel_state import get_self_pp_group
        from sglang.srt.managers.schedule_batch import set_mamba_track_indices_from_reqs

        sidecar_gpu_id = sidecar.gpu_id
        target_gpu_id = self.ps.gpu_id
        sidecar_device = f"cuda:{sidecar_gpu_id}"
        mr = sidecar.model_runner
        side_req_pool = mr.req_to_token_pool
        side_kv_alloc = mr.token_to_kv_pool_allocator
        side_req = state["req"]
        side_batch = state["batch"]
        side_req_idx = int(state["req_idx"])
        old_seq_len = int(state["seq_len"])

        def to_side(t):
            if t is None or not isinstance(t, torch.Tensor):
                return t
            if t.device.type == "cuda":
                t = t.detach().to("cpu")
            return t.to(sidecar_device)

        try:
            torch.cuda.set_device(sidecar_gpu_id)

            accept_len = int(batch_result.accept_lens.reshape(-1)[0].item())
            if accept_len < 1 or accept_len > self.speculative_num_draft_tokens:
                raise RuntimeError(
                    f"unexpected accept_len={accept_len} for CUDA2 persistent shadow"
                )

            # Mirror the four verify candidates into the independent sidecar
            # request row.  For TOPK=1 the accepted path is already the leading
            # contiguous prefix, so new committed length is old + accept_len.
            width = int(self.speculative_num_draft_tokens)
            extend_slots = side_kv_alloc.alloc(width)
            if extend_slots is None:
                raise RuntimeError(
                    f"CUDA2 persistent draft-extend KV reservation failed: need={width}"
                )
            side_req_pool.write(
                (side_req_idx, slice(old_seq_len, old_seq_len + width)),
                extend_slots,
            )
            side_batch.out_cache_loc = extend_slots
            side_batch.seq_lens_cpu = torch.tensor([old_seq_len], dtype=torch.int64)
            side_batch.seq_lens = side_batch.seq_lens_cpu.to(sidecar_device)
            side_batch.seq_lens_sum = old_seq_len
            side_batch.req_pool_indices_cpu = torch.tensor(
                [side_req_idx], dtype=torch.int64
            )
            side_batch.req_pool_indices = side_batch.req_pool_indices_cpu.to(
                sidecar_device
            )
            side_batch.reqs = [side_req]
            side_batch.req_to_token_pool = side_req_pool
            side_batch.token_to_kv_pool_allocator = side_kv_alloc
            side_batch.device = sidecar_device

            # Refresh Mamba ownership/tracking from the persistent CUDA2 Req.
            if hasattr(side_req_pool, "mamba_pool"):
                set_mamba_track_indices_from_reqs(side_batch)
                side_batch._collect_deferred_mamba_cow_and_clear([side_req])

            accept_lens_side = to_side(batch_result.accept_lens.to(torch.int64))
            draft_extend_input = EagleDraftExtendInput(
                hidden_states=to_side(batch_result.logits_output.hidden_states),
                num_correct_drafts=accept_lens_side - 1,
                num_accept_tokens=accept_lens_side,
                num_tokens_per_req=width,
                num_tokens_for_logprob_per_req=width,
            )
            predict = to_side(batch_result.next_token_ids.to(torch.int64))

            with (
                _mtp_sidecar_parallel_context(get_self_pp_group()),
                draft_pp_context(),
                speculative_moe_backend_context(),
                speculative_moe_a2a_backend_context(),
            ):
                side_fb = prepare_for_draft_extend(
                    draft_extend_input,
                    side_batch,
                    predict,
                    width,
                    mr,
                    None,
                    return_hidden_states_before_norm=False,
                )
                side_fb.return_logprob = False
                side_logits = mr.forward(side_fb).logits_output
                torch.cuda.synchronize(sidecar_gpu_id)

            select_index = accept_len - 1
            selected_logits = side_logits.next_token_logits[select_index : select_index + 1]
            selected_hidden = (
                side_logits.hidden_states[select_index : select_index + 1]
                if side_logits.hidden_states is not None
                else None
            )
            side_topk = torch.argmax(selected_logits, dim=-1, keepdim=True).to(
                torch.int64
            )
            side_topk_p = torch.ones_like(side_topk, dtype=torch.float32)
            side_bonus = to_side(batch_result.next_draft_input.bonus_tokens)

            state["draft_input"] = EagleDraftInput(
                topk_p=side_topk_p,
                topk_index=side_topk,
                draft_probs=None,
                hidden_states=selected_hidden,
                bonus_tokens=side_bonus,
                num_tokens_per_req=1,
                num_tokens_for_logprob_per_req=1,
            )
            state["seq_len"] = old_seq_len + accept_len
            side_req.kv_committed_len = state["seq_len"]

            stock_topk = batch_result.next_draft_input.topk_index.detach().to("cpu")
            side_topk_cpu = side_topk.detach().to("cpu")
            topk_match = torch.equal(side_topk_cpu, stock_topk)

            hidden_cos = None
            stock_hidden = batch_result.next_draft_input.hidden_states
            if selected_hidden is not None and stock_hidden is not None:
                a = selected_hidden.detach().float().cpu().reshape(-1)
                b = stock_hidden.detach().float().cpu().reshape(-1)
                if a.shape == b.shape:
                    hidden_cos = torch.nn.functional.cosine_similarity(a, b, dim=0).item()

            logger.info(
                "[MTP-SIDECAR-EXTEND-COMPARE] old_seq=%d accept=%d new_seq=%d "
                "side_topk=%s stock_topk=%s match=%s hidden_cosine=%s",
                old_seq_len,
                accept_len,
                state["seq_len"],
                side_topk_cpu.tolist(),
                stock_topk.tolist(),
                topk_match,
                (f"{hidden_cos:.9f}" if hidden_cos is not None else None),
            )

            # Build the proposal for the NEXT stock draft iteration.  The
            # existing draft() comparator will check it token-for-token.
            self._mtp_sidecar_persistent_proposal()

        except Exception:
            logger.exception("[MTP-SIDECAR-PERSIST] draft-extend FAILED")
            self._mtp_sidecar_persist_state = None
        finally:
            torch.cuda.set_device(target_gpu_id)

'''
    if method_marker not in s:
        raise RuntimeError("_draft_extend_for_decode insertion point not found")
    s = s.replace(method_marker, helpers + method_marker, 1)

    # Run the sidecar mirror only after the stock path has fully materialized
    # next_draft_input, so we can compare its selected top-k/hidden state.
    tail_marker = '''        if self.seed_dsa_topk_from_draft_extend:\n            next_draft_input.dsa_topk_indices = dsa_seed_topk_indices\n\n\nclass EAGLEWorkerV2(BaseSpecWorker):\n'''
    tail_replacement = '''        if self.seed_dsa_topk_from_draft_extend:\n            next_draft_input.dsa_topk_indices = dsa_seed_topk_indices\n\n        # Persistent CUDA2 correctness shadow.  Stock remains authoritative.\n        self._mtp_sidecar_shadow_extend_for_decode(batch, batch_result)\n\n\nclass EAGLEWorkerV2(BaseSpecWorker):\n'''
    if tail_marker not in s:
        raise RuntimeError("stock draft-extend tail insertion point not found")
    s = s.replace(tail_marker, tail_replacement, 1)

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

    backup = eagle_src.with_suffix(eagle_src.suffix + ".before-persistence-shadow")
    if not backup.exists():
        shutil.copy2(eagle_src, backup)

    patch_eagle(eagle_src)
    shutil.copy2(eagle_src, eagle_dst)

    run("python3", "-m", "py_compile", str(eagle_src))
    run("git", "diff", "--check", cwd=repo)
    print("PERSISTENCE SHADOW PATCHED OK")

    if args.commit:
        run("git", "add", str(eagle_dst.relative_to(repo)), cwd=repo)
        status = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=repo)
        if status.returncode != 0:
            run(
                "git",
                "commit",
                "-m",
                "wip: persist CUDA2 MTP shadow across verify",
                cwd=repo,
            )
        run("git", "push", "origin", f"HEAD:{BRANCH}", cwd=repo)
        print("PERSISTENCE SHADOW COMMIT/PUSH OK")


if __name__ == "__main__":
    main()
