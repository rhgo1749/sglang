#!/usr/bin/env python3
"""Repair the authoritative CUDA2 MTP cutover's nested construction contexts.

The outer EAGLEWorkerV2 must establish only the sidecar TP/attention topology
while constructing EagleDraftWorker.  EagleDraftWorker.__init__ already enters
its own draft PP and speculative-MoE contexts.  Entering those contexts again
from the outer worker is invalid (patch_pipeline_parallel_group is explicitly
non-reentrant) and aborts startup with `_PP_STATE_PATCHED`.

This hotfix is intentionally idempotent and repairs all three copies that matter:
  * the host exact-image mount source used by Docker,
  * the fork's patched eagle_worker_v2.py checkpoint,
  * the cutover generator so a clean re-application does not recreate the bug.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

BRANCH = "wip/qwen38-mtp-sidecar-cuda2"

OLD = '''                with (\n                    _mtp_sidecar_parallel_context(get_self_pp_group()),\n                    draft_pp_context(),\n                    speculative_moe_backend_context(),\n                    speculative_moe_a2a_backend_context(),\n                ):\n                    self._draft_worker = EagleDraftWorker(\n'''

NEW = '''                # EagleDraftWorker.__init__ owns draft_pp_context and the\n                # speculative MoE contexts itself.  The PP patcher is deliberately\n                # non-reentrant, so the outer worker must establish only the TP1 /\n                # attention topology required while the sidecar modules are built.\n                with _mtp_sidecar_parallel_context(get_self_pp_group()):\n                    self._draft_worker = EagleDraftWorker(\n'''


def run(*args: str, cwd: Path | None = None) -> None:
    subprocess.run(args, cwd=cwd, check=True)


def patch_one(path: Path, *, required: bool) -> bool:
    if not path.exists():
        if required:
            raise RuntimeError(f"required file not found: {path}")
        print(f"skip missing: {path}")
        return False

    text = path.read_text()
    if NEW in text:
        print(f"already fixed: {path}")
        return False
    if OLD not in text:
        if required:
            raise RuntimeError(f"PP hotfix pattern not found: {path}")
        print(f"no cutover block yet: {path}")
        return False

    count = text.count(OLD)
    if count != 1:
        raise RuntimeError(f"expected one PP cutover block in {path}, found {count}")
    path.write_text(text.replace(OLD, NEW, 1))
    print(f"fixed nested PP context: {path}")
    return True


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    patch_dir = Path.home() / "projects" / "sglang-patches"

    host_eagle = patch_dir / "eagle_worker_v2.sidecar-pool-probe.py"
    repo_eagle = repo / "python/sglang/srt/speculative/eagle_worker_v2.py"
    generator = repo / "tools/qwen38_mtp_sidecar_cutover.py"

    # Current machine already has the cutover installed, so the mounted host
    # source is the authoritative runtime copy and must be repaired.
    changed_host = patch_one(host_eagle, required=True)

    # Keep the fork checkpoint in lockstep with the exact-image runtime copy.
    repo_eagle.write_text(host_eagle.read_text())
    print(f"synced runtime source: {repo_eagle}")

    # Repair the generator as well.  OLD appears literally inside its raw
    # `outer_block`, so the same replacement is valid there.
    changed_generator = patch_one(generator, required=True)

    run("python3", "-m", "py_compile", str(host_eagle))
    run("python3", "-m", "py_compile", str(generator))
    run("git", "diff", "--check", cwd=repo)

    run(
        "git",
        "add",
        str(repo_eagle.relative_to(repo)),
        str(generator.relative_to(repo)),
        cwd=repo,
    )
    status = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=repo)
    if status.returncode != 0:
        run(
            "git",
            "commit",
            "-m",
            "fix: avoid nested PP context in CUDA2 MTP cutover",
            cwd=repo,
        )
        run("git", "push", "origin", f"HEAD:{BRANCH}", cwd=repo)
        print("MTP CUTOVER PP HOTFIX COMMIT/PUSH OK")
    else:
        print("MTP CUTOVER PP HOTFIX ALREADY APPLIED")

    print(
        "MTP CUTOVER PP HOTFIX OK "
        f"host_changed={changed_host} generator_changed={changed_generator}"
    )


if __name__ == "__main__":
    main()
