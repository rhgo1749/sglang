#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pathlib
import subprocess

HOME = pathlib.Path.home()
REPO = HOME / "projects/sglang-fork"
HOST = HOME / "projects/sglang-patches/eagle_worker_v2.sidecar-pool-probe.py"
FORK = REPO / "python/sglang/srt/speculative/eagle_worker_v2.py"

SIDE_OLD = '''                if int(mr.max_total_num_tokens) < 65536:\n                    raise RuntimeError(\n                        f"CUDA2 MTP pool too small for cutover: "\n                        f"{mr.max_total_num_tokens} < 65536"\n                    )\n'''
SIDE_NEW = '''                import os as _mtp_os\n                _mtp_min_pool_tokens = int(\n                    _mtp_os.environ.get(\n                        "SGLANG_MTP_CUTOVER_MIN_POOL_TOKENS", "65536"\n                    )\n                )\n                if int(mr.max_total_num_tokens) < _mtp_min_pool_tokens:\n                    raise RuntimeError(\n                        f"CUDA2 MTP pool too small for cutover: "\n                        f"{mr.max_total_num_tokens} < {_mtp_min_pool_tokens}"\n                    )\n'''

TARGET_OLD = '''            if target_tokens < 65536:\n                raise RuntimeError(\n                    f"MTP cutover target KV pool too small: "\n                    f"{target_tokens} < 65536"\n                )\n'''
TARGET_NEW = '''            import os as _mtp_os\n            _mtp_min_pool_tokens = int(\n                _mtp_os.environ.get(\n                    "SGLANG_MTP_CUTOVER_MIN_POOL_TOKENS", "65536"\n                )\n            )\n            if target_tokens < _mtp_min_pool_tokens:\n                raise RuntimeError(\n                    f"MTP cutover target KV pool too small: "\n                    f"{target_tokens} < {_mtp_min_pool_tokens}"\n                )\n'''


def patch(path: pathlib.Path) -> bool:
    text = path.read_text()
    changed = False
    if SIDE_NEW not in text:
        if SIDE_OLD not in text:
            raise RuntimeError(f"side pool gate patch point not found: {path}")
        text = text.replace(SIDE_OLD, SIDE_NEW, 1)
        changed = True
    if TARGET_NEW not in text:
        if TARGET_OLD not in text:
            raise RuntimeError(f"target pool gate patch point not found: {path}")
        text = text.replace(TARGET_OLD, TARGET_NEW, 1)
        changed = True
    if changed:
        path.write_text(text)
        print(f"fixed configurable pool gate: {path}")
    else:
        print(f"pool gate already configurable: {path}")
    return changed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--commit", action="store_true")
    args = ap.parse_args()

    host_changed = patch(HOST)
    fork_changed = patch(FORK)
    print(
        "MTP CUTOVER POOL GATE HOTFIX OK "
        f"host_changed={host_changed} fork_changed={fork_changed}"
    )

    if args.commit and fork_changed:
        subprocess.run(
            ["git", "add", str(FORK.relative_to(REPO))], cwd=REPO, check=True
        )
        subprocess.run(
            ["git", "commit", "-m", "fix: make MTP cutover pool gate configurable"],
            cwd=REPO,
            check=True,
        )
        subprocess.run(
            ["git", "push", "origin", "HEAD:wip/qwen38-mtp-sidecar-cuda2"],
            cwd=REPO,
            check=True,
        )
        print("MTP CUTOVER POOL GATE COMMIT/PUSH OK")


if __name__ == "__main__":
    main()
