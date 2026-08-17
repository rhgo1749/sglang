#!/usr/bin/env python3
"""Make the CUDA2 sidecar multi-step DraftBackendFactory call image-compatible.

The WIP branch vendors exact-image copies of eagle_worker_v2.py/qwen3_5_mtp.py,
while the surrounding fork can be newer than lmsysorg/sglang:qwen38-27b.  In
particular DraftBackendFactory changed constructor shape across revisions:

  newer tree: DraftBackendFactory(draft_model_runner, topk,
                                  speculative_num_steps, ...)
  exact image: may additionally require server_args

Do not guess the version again.  Patch the sidecar to inspect the constructor
at runtime and pass server_args only when that parameter actually exists.
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

    variants = [
        '''                    _side_factory = DraftBackendFactory(\n                        self.server_args,\n                        mr,\n                        self.topk,\n                        self.speculative_num_steps,\n                        seed_dsa_topk_from_draft_extend=False,\n                    )\n''',
        '''                    _side_factory = DraftBackendFactory(\n                        mr,\n                        self.topk,\n                        self.speculative_num_steps,\n                        seed_dsa_topk_from_draft_extend=False,\n                    )\n''',
    ]

    adaptive = '''                    # This WIP file is mounted into an exact Docker image while\n                    # the fork around it can be a newer SGLang revision.\n                    # DraftBackendFactory changed constructor shape between\n                    # those revisions, so resolve it from the runtime class\n                    # instead of hard-coding either ABI.\n                    import inspect as _inspect\n\n                    _factory_params = _inspect.signature(\n                        DraftBackendFactory.__init__\n                    ).parameters\n                    _factory_kwargs = {\n                        "draft_model_runner": mr,\n                        "topk": self.topk,\n                        "speculative_num_steps": self.speculative_num_steps,\n                        "seed_dsa_topk_from_draft_extend": False,\n                    }\n                    if "server_args" in _factory_params:\n                        _factory_kwargs["server_args"] = self.server_args\n\n                    logger.info(\n                        "[MTP-SIDECAR-FACTORY] DraftBackendFactory params=%s server_args=%s",\n                        list(_factory_params),\n                        "server_args" in _factory_params,\n                    )\n                    _side_factory = DraftBackendFactory(**_factory_kwargs)\n'''

    if adaptive in s:
        print("adaptive constructor call already installed")
    else:
        for old in variants:
            if old in s:
                s = s.replace(old, adaptive, 1)
                break
        else:
            raise RuntimeError("DraftBackendFactory sidecar constructor call not found")

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

    backup = eagle_src.with_suffix(eagle_src.suffix + ".before-adaptive-factory-fix")
    if not backup.exists():
        shutil.copy2(eagle_src, backup)

    patch_eagle(eagle_src)
    shutil.copy2(eagle_src, eagle_dst)

    run("python3", "-m", "py_compile", str(eagle_src))
    run("git", "diff", "--check", cwd=repo)
    print("MULTISTEP ADAPTIVE FACTORY FIX OK")

    if args.commit:
        run("git", "add", str(eagle_dst.relative_to(repo)), cwd=repo)
        status = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=repo)
        if status.returncode != 0:
            run(
                "git",
                "commit",
                "-m",
                "wip: adapt CUDA2 multistep backend factory ABI",
                cwd=repo,
            )
        run("git", "push", "origin", f"HEAD:{BRANCH}", cwd=repo)
        print("MULTISTEP ADAPTIVE FACTORY FIX COMMIT/PUSH OK")


if __name__ == "__main__":
    main()
