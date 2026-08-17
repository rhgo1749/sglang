#!/usr/bin/env python3
"""Add a one-shot CUDA2 sidecar-vs-colocated MTP correctness comparator.

The sidecar shadow prefill already runs before the authoritative colocated
EAGLE/MTP prefill on the same ScheduleBatch.  Save the sidecar's final logits
and hidden state on CPU, then compare them with the colocated draft result after
its forward.  CPU staging keeps the diagnostic independent of CUDA P2P and of
the target TP2 NCCL group.
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

    if "[MTP-SIDECAR-COMPARE]" in s:
        return

    stash_marker = '''            probe_token = int(\n                torch.argmax(\n                    side_logits.next_token_logits[-1],\n                    dim=-1,\n                ).item()\n            )\n\n            logger.info(\n                "[MTP-SIDECAR-SHADOW] eager prefill SUCCESS "\n'''

    stash_replacement = '''            probe_token = int(\n                torch.argmax(\n                    side_logits.next_token_logits[-1],\n                    dim=-1,\n                ).item()\n            )\n\n            # Save a tiny one-shot correctness snapshot on host memory.  The\n            # authoritative colocated draft runs immediately after this helper\n            # on the same logical prefill.  Host staging deliberately avoids\n            # CUDA2->CUDA0 P2P/IPC assumptions while we validate equivalence.\n            self._mtp_sidecar_shadow_compare = {\n                "logits": side_logits.next_token_logits.detach().float().cpu(),\n                "hidden": (\n                    side_logits.hidden_states.detach().float().cpu()\n                    if side_logits.hidden_states is not None\n                    else None\n                ),\n                "argmax": probe_token,\n            }\n\n            logger.info(\n                "[MTP-SIDECAR-SHADOW] eager prefill SUCCESS "\n'''

    if stash_marker not in s:
        raise RuntimeError("sidecar success stash insertion point not found")
    s = s.replace(stash_marker, stash_replacement, 1)

    compare_marker = '''        maybe_detect_nan(logits_output.next_token_logits, "draft_extend_for_prefill")\n        maybe_detect_inf(logits_output.next_token_logits, "draft_extend_for_prefill")\n\n        prefill_dsa_topk = None\n'''

    compare_replacement = '''        maybe_detect_nan(logits_output.next_token_logits, "draft_extend_for_prefill")\n        maybe_detect_inf(logits_output.next_token_logits, "draft_extend_for_prefill")\n\n        # One-shot numerical comparison between the CUDA2 TP1 sidecar and the\n        # authoritative colocated draft.  Both consumed the same shifted token\n        # stream and target hidden states.  MTP uses top-k=1 here, so top1\n        # agreement is the primary semantic check; cosine / absolute deltas\n        # expose subtler loader or TP-sharding mistakes before replacement.\n        _side_cmp = getattr(self, "_mtp_sidecar_shadow_compare", None)\n        if _side_cmp is not None:\n            try:\n                _stock_logits = logits_output.next_token_logits.detach().float().cpu()\n                _side_logits_cpu = _side_cmp["logits"]\n\n                if _stock_logits.shape != _side_logits_cpu.shape:\n                    logger.error(\n                        "[MTP-SIDECAR-COMPARE] SHAPE_MISMATCH side=%s stock=%s",\n                        tuple(_side_logits_cpu.shape),\n                        tuple(_stock_logits.shape),\n                    )\n                else:\n                    _delta = (_side_logits_cpu - _stock_logits).abs()\n                    _side_flat = _side_logits_cpu.reshape(-1)\n                    _stock_flat = _stock_logits.reshape(-1)\n                    _logit_cos = torch.nn.functional.cosine_similarity(\n                        _side_flat, _stock_flat, dim=0\n                    ).item()\n\n                    _side_argmax = int(torch.argmax(_side_logits_cpu[-1]).item())\n                    _stock_argmax = int(torch.argmax(_stock_logits[-1]).item())\n                    _k = min(5, _stock_logits.shape[-1])\n                    _side_top5 = set(\n                        torch.topk(_side_logits_cpu[-1], k=_k).indices.tolist()\n                    )\n                    _stock_top5 = set(\n                        torch.topk(_stock_logits[-1], k=_k).indices.tolist()\n                    )\n\n                    _hidden_cos = None\n                    _hidden_max_abs = None\n                    _side_hidden = _side_cmp.get("hidden")\n                    _stock_hidden = (\n                        logits_output.hidden_states.detach().float().cpu()\n                        if logits_output.hidden_states is not None\n                        else None\n                    )\n                    if (\n                        _side_hidden is not None\n                        and _stock_hidden is not None\n                        and _side_hidden.shape == _stock_hidden.shape\n                    ):\n                        _hidden_cos = torch.nn.functional.cosine_similarity(\n                            _side_hidden.reshape(-1),\n                            _stock_hidden.reshape(-1),\n                            dim=0,\n                        ).item()\n                        _hidden_max_abs = (\n                            (_side_hidden - _stock_hidden).abs().max().item()\n                        )\n\n                    logger.info(\n                        "[MTP-SIDECAR-COMPARE] "\n                        "shape=%s side_argmax=%d stock_argmax=%d top1_match=%s "\n                        "top5_overlap=%d logit_cosine=%.9f "\n                        "max_abs=%.7g mean_abs=%.7g "\n                        "hidden_cosine=%s hidden_max_abs=%s",\n                        tuple(_stock_logits.shape),\n                        _side_argmax,\n                        _stock_argmax,\n                        _side_argmax == _stock_argmax,\n                        len(_side_top5 & _stock_top5),\n                        _logit_cos,\n                        _delta.max().item(),\n                        _delta.mean().item(),\n                        (\n                            f"{_hidden_cos:.9f}"\n                            if _hidden_cos is not None\n                            else None\n                        ),\n                        (\n                            f"{_hidden_max_abs:.7g}"\n                            if _hidden_max_abs is not None\n                            else None\n                        ),\n                    )\n            finally:\n                self._mtp_sidecar_shadow_compare = None\n\n        prefill_dsa_topk = None\n'''

    if compare_marker not in s:
        raise RuntimeError("colocated draft compare insertion point not found")
    s = s.replace(compare_marker, compare_replacement, 1)
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

    backup = eagle_src.with_suffix(eagle_src.suffix + ".before-compare")
    if not backup.exists():
        shutil.copy2(eagle_src, backup)

    patch_eagle(eagle_src)
    shutil.copy2(eagle_src, eagle_dst)

    run("python3", "-m", "py_compile", str(eagle_src))
    run("git", "diff", "--check", cwd=repo)
    print("COMPARE PATCHED OK")

    if args.commit:
        run("git", "add", str(eagle_dst.relative_to(repo)), cwd=repo)
        status = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=repo)
        if status.returncode != 0:
            run(
                "git",
                "commit",
                "-m",
                "wip: compare CUDA2 sidecar and colocated MTP prefill",
                cwd=repo,
            )
        run("git", "push", "origin", f"HEAD:{BRANCH}", cwd=repo)
        print("COMPARE COMMIT/PUSH OK")


if __name__ == "__main__":
    main()
