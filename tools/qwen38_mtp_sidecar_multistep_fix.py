#!/usr/bin/env python3
"""Make the CUDA2 sidecar multi-step draft probe image-compatible.

This WIP branch vendors exact-image copies of eagle_worker_v2.py/qwen3_5_mtp.py,
while the surrounding fork can be newer than lmsysorg/sglang:qwen38-27b.
Two compatibility/details matter for the shadow multi-step probe:

1. DraftBackendFactory changed constructor shape across revisions.  Resolve its
   runtime signature and pass server_args only when that exact image requires it.
2. The real EAGLE draft() path runs on a DECODE batch.  The shadow probe starts
   immediately after prefill, so reusing the prefill ScheduleBatch mode/input
   makes FlashInferMultiStepDraftBackend incorrectly enter its prefill planner
   (whose decode-only child backends intentionally have no indices_updater_prefill).
   Convert the isolated sidecar batch to a one-token DECODE view before
   prepare_for_draft().
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

    # ---- DraftBackendFactory ABI compatibility ----
    variants = [
        '''                    _side_factory = DraftBackendFactory(\n                        self.server_args,\n                        mr,\n                        self.topk,\n                        self.speculative_num_steps,\n                        seed_dsa_topk_from_draft_extend=False,\n                    )\n''',
        '''                    _side_factory = DraftBackendFactory(\n                        mr,\n                        self.topk,\n                        self.speculative_num_steps,\n                        seed_dsa_topk_from_draft_extend=False,\n                    )\n''',
    ]

    adaptive = '''                    # This WIP file is mounted into an exact Docker image while\n                    # the fork around it can be a newer SGLang revision.\n                    # DraftBackendFactory changed constructor shape between\n                    # those revisions, so resolve it from the runtime class\n                    # instead of hard-coding either ABI.\n                    import inspect as _inspect\n\n                    _factory_params = _inspect.signature(\n                        DraftBackendFactory.__init__\n                    ).parameters\n                    _factory_kwargs = {\n                        "draft_model_runner": mr,\n                        "topk": self.topk,\n                        "speculative_num_steps": self.speculative_num_steps,\n                        "seed_dsa_topk_from_draft_extend": False,\n                    }\n                    if "server_args" in _factory_params:\n                        _factory_kwargs["server_args"] = self.server_args\n\n                    logger.info(\n                        "[MTP-SIDECAR-FACTORY] DraftBackendFactory params=%s server_args=%s",\n                        list(_factory_params),\n                        "server_args" in _factory_params,\n                    )\n                    _side_factory = DraftBackendFactory(**_factory_kwargs)\n'''

    if adaptive not in s:
        for old in variants:
            if old in s:
                s = s.replace(old, adaptive, 1)
                break
        else:
            raise RuntimeError("DraftBackendFactory sidecar constructor call not found")

    # ---- Convert the post-prefill shadow batch to the same DECODE view that
    # stock EagleDraftWorker.draft() receives.  prepare_for_draft() itself does
    # not rewrite ForwardMode or batch.input_ids; carrying EXTEND + 53 prompt
    # tokens into FlashInferMultiStepDraftBackend makes its decode-only child
    # backend take the prefill metadata path and fail on indices_updater_prefill.
    old_mode = '''            side_batch.spec_info = _side_draft_input\n            side_batch.forward_mode = batch.forward_mode\n            side_batch.seq_lens_cpu = torch.tensor([seq_len], dtype=torch.int64)\n            side_batch.seq_lens = side_batch.seq_lens_cpu.to(sidecar_device)\n            side_batch.seq_lens_sum = seq_len\n'''
    new_mode = '''            side_batch.spec_info = _side_draft_input\n            from sglang.srt.model_executor.forward_batch_info import ForwardMode as _ForwardMode\n\n            side_batch.forward_mode = _ForwardMode.DECODE\n            side_batch.input_ids = _side_initial_token.reshape(-1)\n            side_batch.seq_lens_cpu = torch.tensor([seq_len], dtype=torch.int64)\n            side_batch.seq_lens = side_batch.seq_lens_cpu.to(sidecar_device)\n            side_batch.seq_lens_sum = seq_len\n            logger.info(\n                "[MTP-SIDECAR-DECODE-VIEW] mode=%s input_shape=%s seq_len=%d",\n                side_batch.forward_mode,\n                tuple(side_batch.input_ids.shape),\n                seq_len,\n            )\n'''

    if old_mode in s:
        s = s.replace(old_mode, new_mode, 1)
    elif new_mode not in s:
        raise RuntimeError("sidecar post-prefill decode-view insertion point not found")

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

    backup = eagle_src.with_suffix(eagle_src.suffix + ".before-adaptive-factory-decode-fix")
    if not backup.exists():
        shutil.copy2(eagle_src, backup)

    patch_eagle(eagle_src)
    shutil.copy2(eagle_src, eagle_dst)

    run("python3", "-m", "py_compile", str(eagle_src))
    run("git", "diff", "--check", cwd=repo)
    print("MULTISTEP FACTORY+DECODE FIX OK")

    if args.commit:
        run("git", "add", str(eagle_dst.relative_to(repo)), cwd=repo)
        status = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=repo)
        if status.returncode != 0:
            run(
                "git",
                "commit",
                "-m",
                "wip: fix CUDA2 multistep decode view",
                cwd=repo,
            )
        run("git", "push", "origin", f"HEAD:{BRANCH}", cwd=repo)
        print("MULTISTEP FACTORY+DECODE COMMIT/PUSH OK")


if __name__ == "__main__":
    main()
