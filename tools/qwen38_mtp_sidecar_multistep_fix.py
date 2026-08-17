#!/usr/bin/env python3
"""Fix the CUDA2 sidecar multi-step DraftBackendFactory constructor call.

The exact-image DraftBackendFactory signature is:
    DraftBackendFactory(draft_model_runner, topk, speculative_num_steps,
                        seed_dsa_topk_from_draft_extend=False)

The first multi-step probe accidentally passed server_args as an extra leading
positional argument, so the shadow prefill failed before the proposal chain
could start.  This patch only corrects that constructor call and leaves the
already-proven prefill/resource/topology path unchanged.
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

    old = '''                    _side_factory = DraftBackendFactory(\n                        self.server_args,\n                        mr,\n                        self.topk,\n                        self.speculative_num_steps,\n                        seed_dsa_topk_from_draft_extend=False,\n                    )\n'''
    new = '''                    _side_factory = DraftBackendFactory(\n                        mr,\n                        self.topk,\n                        self.speculative_num_steps,\n                        seed_dsa_topk_from_draft_extend=False,\n                    )\n'''

    if old in s:
        s = s.replace(old, new, 1)
    elif new in s:
        print("constructor call already fixed")
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

    backup = eagle_src.with_suffix(eagle_src.suffix + ".before-multistep-factory-fix")
    if not backup.exists():
        shutil.copy2(eagle_src, backup)

    patch_eagle(eagle_src)
    shutil.copy2(eagle_src, eagle_dst)

    run("python3", "-m", "py_compile", str(eagle_src))
    run("git", "diff", "--check", cwd=repo)
    print("MULTISTEP FACTORY FIX OK")

    if args.commit:
        run("git", "add", str(eagle_dst.relative_to(repo)), cwd=repo)
        status = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=repo)
        if status.returncode != 0:
            run(
                "git",
                "commit",
                "-m",
                "wip: fix CUDA2 multistep backend factory call",
                cwd=repo,
            )
        run("git", "push", "origin", f"HEAD:{BRANCH}", cwd=repo)
        print("MULTISTEP FACTORY FIX COMMIT/PUSH OK")


if __name__ == "__main__":
    main()
