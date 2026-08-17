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
SIDE_PREV = '''                import os as _mtp_os\n                _mtp_default_min_pool = (\n                    32768\n                    if int(get_schedule().max_running_requests or 1) > 1\n                    else 65536\n                )\n                _mtp_min_pool_tokens = int(\n                    _mtp_os.environ.get(\n                        "SGLANG_MTP_CUTOVER_MIN_POOL_TOKENS",\n                        str(_mtp_default_min_pool),\n                    )\n                )\n                if int(mr.max_total_num_tokens) < _mtp_min_pool_tokens:\n                    raise RuntimeError(\n                        f"CUDA2 MTP pool too small for cutover: "\n                        f"{mr.max_total_num_tokens} < {_mtp_min_pool_tokens}"\n                    )\n'''
SIDE_OLDER = '''                import os as _mtp_os\n                _mtp_min_pool_tokens = int(\n                    _mtp_os.environ.get(\n                        "SGLANG_MTP_CUTOVER_MIN_POOL_TOKENS", "65536"\n                    )\n                )\n                if int(mr.max_total_num_tokens) < _mtp_min_pool_tokens:\n                    raise RuntimeError(\n                        f"CUDA2 MTP pool too small for cutover: "\n                        f"{mr.max_total_num_tokens} < {_mtp_min_pool_tokens}"\n                    )\n'''
SIDE_NEW = '''                import os as _mtp_os\n                _mtp_default_min_pool = (\n                    32768\n                    if int(getattr(self.server_args, "max_running_requests", 1) or 1) > 1\n                    else 65536\n                )\n                _mtp_min_pool_tokens = int(\n                    _mtp_os.environ.get(\n                        "SGLANG_MTP_CUTOVER_MIN_POOL_TOKENS",\n                        str(_mtp_default_min_pool),\n                    )\n                )\n                if int(mr.max_total_num_tokens) < _mtp_min_pool_tokens:\n                    raise RuntimeError(\n                        f"CUDA2 MTP pool too small for cutover: "\n                        f"{mr.max_total_num_tokens} < {_mtp_min_pool_tokens}"\n                    )\n'''

TARGET_OLD = '''            if target_tokens < 65536:\n                raise RuntimeError(\n                    f"MTP cutover target KV pool too small: "\n                    f"{target_tokens} < 65536"\n                )\n'''
TARGET_PREV = '''            import os as _mtp_os\n            _mtp_default_min_pool = (\n                32768\n                if int(get_schedule().max_running_requests or 1) > 1\n                else 65536\n            )\n            _mtp_min_pool_tokens = int(\n                _mtp_os.environ.get(\n                    "SGLANG_MTP_CUTOVER_MIN_POOL_TOKENS",\n                    str(_mtp_default_min_pool),\n                )\n            )\n            if target_tokens < _mtp_min_pool_tokens:\n                raise RuntimeError(\n                    f"MTP cutover target KV pool too small: "\n                    f"{target_tokens} < {_mtp_min_pool_tokens}"\n                )\n'''
TARGET_OLDER = '''            import os as _mtp_os\n            _mtp_min_pool_tokens = int(\n                _mtp_os.environ.get(\n                    "SGLANG_MTP_CUTOVER_MIN_POOL_TOKENS", "65536"\n                )\n            )\n            if target_tokens < _mtp_min_pool_tokens:\n                raise RuntimeError(\n                    f"MTP cutover target KV pool too small: "\n                    f"{target_tokens} < {_mtp_min_pool_tokens}"\n                )\n'''
TARGET_NEW = '''            import os as _mtp_os\n            _mtp_default_min_pool = (\n                32768\n                if int(getattr(self.server_args, "max_running_requests", 1) or 1) > 1\n                else 65536\n            )\n            _mtp_min_pool_tokens = int(\n                _mtp_os.environ.get(\n                    "SGLANG_MTP_CUTOVER_MIN_POOL_TOKENS",\n                    str(_mtp_default_min_pool),\n                )\n            )\n            if target_tokens < _mtp_min_pool_tokens:\n                raise RuntimeError(\n                    f"MTP cutover target KV pool too small: "\n                    f"{target_tokens} < {_mtp_min_pool_tokens}"\n                )\n'''


def replace_variant(text: str, new: str, variants: tuple[str, ...], label: str) -> tuple[str, bool]:
    if new in text:
        return text, False
    for old in variants:
        if old in text:
            return text.replace(old, new, 1), True
    raise RuntimeError(f"{label} patch point not found")


def patch(path: pathlib.Path) -> bool:
    text = path.read_text()
    text, side_changed = replace_variant(
        text, SIDE_NEW, (SIDE_PREV, SIDE_OLDER, SIDE_OLD), "side pool gate"
    )
    text, target_changed = replace_variant(
        text, TARGET_NEW, (TARGET_PREV, TARGET_OLDER, TARGET_OLD), "target pool gate"
    )
    changed = side_changed or target_changed
    if changed:
        path.write_text(text)
        subprocess.run(["python3", "-m", "py_compile", str(path)], check=True)
        print(f"fixed parallel-aware pool gate: {path}")
    else:
        print(f"parallel-aware pool gate already installed: {path}")
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
        diff = subprocess.run(
            ["git", "diff", "--cached", "--quiet"], cwd=REPO, check=False
        )
        if diff.returncode != 0:
            subprocess.run(
                ["git", "commit", "-m", "fix: make MTP pool gate self-contained"],
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
