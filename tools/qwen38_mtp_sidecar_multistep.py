#!/usr/bin/env python3
"""Add a one-shot CUDA2 TP1 MTP multi-step draft shadow + proposal comparator.

This is the final shadow gate before replacing the colocated draft.  The
existing CUDA2 prefill probe already proves model/pool/backend correctness.
This patch extends that same isolated sidecar request through the actual EAGLE
TOPK=1 draft chain (num_steps=3 in the current launch), stashes the raw proposal
on CPU, and compares it with the colocated draft's raw proposal before target
verify.

The authoritative colocated path is left untouched.  Any sidecar failure is
logged and serving continues through the stock draft.
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
    if "[MTP-SIDECAR-MULTISTEP]" in s:
        return

    # Insert the multi-step run immediately after the sidecar prefill forward,
    # while side_req/side_batch/forward_batch and the sidecar KV/Mamba state are
    # still live and on CUDA2.
    marker = '''            probe_token = int(\n                torch.argmax(\n                    side_logits.next_token_logits[-1],\n                    dim=-1,\n                ).item()\n            )\n\n            # Save a tiny one-shot correctness snapshot on host memory.  The\n'''

    replacement = '''            probe_token = int(\n                torch.argmax(\n                    side_logits.next_token_logits[-1],\n                    dim=-1,\n                ).item()\n            )\n\n            # Exercise the real TOPK=1 EAGLE multi-step chain on CUDA2 before\n            # the stock draft runs.  Reserve the sidecar's future draft KV slots\n            # explicitly: prepare_for_draft() reads them from req_to_token at\n            # [seq_len : seq_len + topk*num_steps].  The scheduler normally\n            # pre-populates those slots for the colocated draft; our independent\n            # sidecar pool must do that itself.\n            if self.topk != 1:\n                raise RuntimeError(\n                    f"CUDA2 multistep shadow currently requires topk=1, got {self.topk}"\n                )\n\n            _num_steps = int(self.speculative_num_steps)\n            _future_slots = side_kv_alloc.alloc(_num_steps)\n            if _future_slots is None:\n                raise RuntimeError(\n                    f"CUDA2 sidecar draft KV reservation failed: need={_num_steps}"\n                )\n            side_req_pool.write(\n                (side_req_idx, slice(seq_len, seq_len + _num_steps)),\n                _future_slots,\n            )\n\n            _side_initial_token = torch.argmax(\n                side_logits.next_token_logits, dim=-1, keepdim=True\n            ).to(torch.int64)\n            _side_draft_input = EagleDraftInput(\n                topk_p=torch.ones_like(_side_initial_token, dtype=torch.float32),\n                topk_index=_side_initial_token,\n                draft_probs=None,\n                hidden_states=side_logits.hidden_states,\n                bonus_tokens=_to_sidecar(next_token_ids),\n                num_tokens_per_req=1,\n                num_tokens_for_logprob_per_req=1,\n            )\n\n            # prepare_for_draft mutates the ScheduleBatch view only.  Reuse the\n            # already-isolated side_batch, whose request row and pools belong to\n            # CUDA2, and create a sidecar-specific multi-step attention backend\n            # once.  Do not borrow the colocated draft backend: it owns CUDA0\n            # buffers and TP2 construction state.\n            side_batch.spec_info = _side_draft_input\n            side_batch.forward_mode = batch.forward_mode\n            side_batch.seq_lens_cpu = torch.tensor([seq_len], dtype=torch.int64)\n            side_batch.seq_lens = side_batch.seq_lens_cpu.to(sidecar_device)\n            side_batch.seq_lens_sum = seq_len\n\n            with (\n                _mtp_sidecar_parallel_context(get_self_pp_group()),\n                draft_pp_context(),\n                speculative_moe_backend_context(),\n                speculative_moe_a2a_backend_context(),\n            ):\n                if not hasattr(self, "_mtp_sidecar_draft_attn_backend"):\n                    _side_factory = DraftBackendFactory(\n                        self.server_args,\n                        mr,\n                        self.topk,\n                        self.speculative_num_steps,\n                        seed_dsa_topk_from_draft_extend=False,\n                    )\n                    self._mtp_sidecar_draft_attn_backend = (\n                        _side_factory.create_decode_backend()\n                    )\n\n                _side_fb, _ = prepare_for_draft(\n                    _side_draft_input,\n                    side_req_pool,\n                    side_batch,\n                    None,\n                    mr,\n                    self.topk,\n                    self.speculative_num_steps,\n                )\n                _side_fb.return_logprob = False\n                _side_attn = self._mtp_sidecar_draft_attn_backend\n                if self.speculative_num_steps > 1:\n                    _side_attn.init_forward_metadata(_side_fb)\n                    _side_fb.mark_forward_metadata_ready()\n\n                _out_cache = per_step_draft_out_cache_loc(\n                    _side_fb.out_cache_loc,\n                    _side_fb.batch_size,\n                    self.topk,\n                    self.speculative_num_steps,\n                )\n                _cur_token = _side_initial_token\n                _cur_hidden = side_logits.hidden_states\n                _proposal = [_cur_token.reshape(-1)]\n\n                # num_steps=3 means two actual MTP forwards after the initial\n                # token from prefill logits.  This mirrors draft_forward's\n                # topk1 fast path without touching the stock worker's state.\n                for _i in range(self.speculative_num_steps - 1):\n                    _side_fb.input_ids = _cur_token.reshape(-1)\n                    _side_fb.out_cache_loc = _out_cache[_i]\n                    _side_fb.spec_info.hidden_states = _cur_hidden\n                    with forward_context(\n                        ForwardContext(attn_backend=_side_attn.attn_backends[_i])\n                    ):\n                        _step_logits = mr.forward(_side_fb).logits_output\n                    torch.cuda.synchronize(sidecar_gpu_id)\n                    _cur_token = torch.argmax(\n                        _step_logits.next_token_logits, dim=-1, keepdim=True\n                    ).to(torch.int64)\n                    _proposal.append(_cur_token.reshape(-1))\n                    _cur_hidden = _step_logits.hidden_states\n                    _side_fb.positions.add_(1)\n\n            self._mtp_sidecar_multistep_proposal = (\n                torch.stack(_proposal, dim=1).detach().cpu()\n            )\n            logger.info(\n                "[MTP-SIDECAR-MULTISTEP] CUDA%d proposal=%s",\n                sidecar_gpu_id,\n                self._mtp_sidecar_multistep_proposal.tolist(),\n            )\n\n            # Save a tiny one-shot correctness snapshot on host memory.  The\n'''

    if marker not in s:
        raise RuntimeError("sidecar multistep insertion point not found")
    s = s.replace(marker, replacement, 1)

    # Compare the raw proposal returned by the authoritative draft_forward.
    marker2 = '''        return build_eagle_verify_input(\n            batch,\n            draft_input,\n            parent_list,\n'''
    replacement2 = '''        _side_proposal = getattr(\n            self, "_mtp_sidecar_multistep_proposal", None\n        )\n        if _side_proposal is not None:\n            try:\n                _stock_proposal = draft_tokens.detach().to("cpu")\n                _same_shape = tuple(_side_proposal.shape) == tuple(_stock_proposal.shape)\n                _exact = bool(\n                    _same_shape and torch.equal(_side_proposal, _stock_proposal)\n                )\n                _matches = (\n                    int((_side_proposal == _stock_proposal).sum().item())\n                    if _same_shape\n                    else -1\n                )\n                logger.info(\n                    "[MTP-SIDECAR-PROPOSAL] side=%s stock=%s shape_match=%s "\n                    "token_matches=%d/%d exact=%s",\n                    _side_proposal.tolist(),\n                    _stock_proposal.tolist(),\n                    _same_shape,\n                    _matches,\n                    _stock_proposal.numel(),\n                    _exact,\n                )\n            finally:\n                self._mtp_sidecar_multistep_proposal = None\n\n        return build_eagle_verify_input(\n            batch,\n            draft_input,\n            parent_list,\n'''
    if marker2 not in s:
        raise RuntimeError("stock draft proposal compare insertion point not found")
    s = s.replace(marker2, replacement2, 1)
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

    backup = eagle_src.with_suffix(eagle_src.suffix + ".before-multistep")
    if not backup.exists():
        shutil.copy2(eagle_src, backup)

    patch_eagle(eagle_src)
    shutil.copy2(eagle_src, eagle_dst)
    run("python3", "-m", "py_compile", str(eagle_src))
    run("git", "diff", "--check", cwd=repo)
    print("MULTISTEP PATCHED OK")

    if args.commit:
        run("git", "add", str(eagle_dst.relative_to(repo)), cwd=repo)
        status = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=repo)
        if status.returncode != 0:
            run(
                "git",
                "commit",
                "-m",
                "wip: shadow CUDA2 MTP multistep proposal",
                cwd=repo,
            )
        run("git", "push", "origin", f"HEAD:{BRANCH}", cwd=repo)
        print("MULTISTEP COMMIT/PUSH OK")


if __name__ == "__main__":
    main()
