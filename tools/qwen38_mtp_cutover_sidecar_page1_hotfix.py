#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pathlib
import subprocess

HOME = pathlib.Path.home()
REPO = HOME / "projects/sglang-fork"
HOST = HOME / "projects/sglang-patches/eagle_worker_v2.sidecar-pool-probe.py"
FORK = REPO / "python/sglang/srt/speculative/eagle_worker_v2.py"

IMPORT_OLD = '''from sglang.srt.runtime_context import (\n    get_context,\n    get_exec,\n    get_model,\n    get_parallel,\n    get_spec,\n)\n'''
IMPORT_NEW = '''from sglang.srt.runtime_context import (\n    get_context,\n    get_exec,\n    get_model,\n    get_parallel,\n    get_schedule,\n    get_spec,\n)\n'''

CTX_OLD = '''            get_context().resources.override(\n                buffers=_MTP_SIDECAR_BUFFERS,\n                streams=_MTP_SIDECAR_STREAMS,\n            ),\n            get_parallel().override(\n'''
CTX_NEW = '''            get_context().resources.override(\n                buffers=_MTP_SIDECAR_BUFFERS,\n                streams=_MTP_SIDECAR_STREAMS,\n            ),\n            # The target may be forced to page_size=64 by TRTLLM-MHA/NVFP4.\n            # CUDA2 owns an independent FlashInfer KV pool and the cutover\n            # helpers allocate exact token spans (including 3/4-token draft\n            # tails), so keep the sidecar on the token allocator (page_size=1).\n            # Otherwise PagedTokenToKVPoolAllocator.alloc() floors non-page-\n            # aligned sizes and silently returns too few slots.\n            get_schedule().override(page_size=1),\n            get_parallel().override(\n'''

# This block predates the NVFP4/BF16 draft split.  Use it only when installing
# page isolation into an older cutover.  Once the page-size assignment exists,
# the draft KV dtype is deliberately irrelevant to this hotfix.
ARGS_OLD_FP8 = '''                _side_server_args = server_args\n                if getattr(server_args, "kv_cache_dtype", None) == "nvfp4":\n                    _side_server_args = _mtp_copy.copy(server_args)\n                    object.__setattr__(\n                        _side_server_args, "kv_cache_dtype", "fp8_e4m3"\n                    )\n'''
ARGS_OLD_BF16 = '''                _side_server_args = server_args\n                if getattr(server_args, "kv_cache_dtype", None) == "nvfp4":\n                    _side_server_args = _mtp_copy.copy(server_args)\n                    object.__setattr__(\n                        _side_server_args, "kv_cache_dtype", "bfloat16"\n                    )\n'''
ARGS_NEW_FP8 = '''                # CUDA2 is a fully independent draft runtime.  Give it a\n                # private args view even when only page size differs, so code\n                # that still reads server_args.page_size agrees with the scoped\n                # schedule bag used while the sidecar is built/run.\n                _side_server_args = _mtp_copy.copy(server_args)\n                object.__setattr__(_side_server_args, "page_size", 1)\n                if getattr(server_args, "kv_cache_dtype", None) == "nvfp4":\n                    object.__setattr__(\n                        _side_server_args, "kv_cache_dtype", "fp8_e4m3"\n                    )\n'''
ARGS_NEW_BF16 = '''                # CUDA2 is a fully independent draft runtime.  Give it a\n                # private args view even when only page size differs, so code\n                # that still reads server_args.page_size agrees with the scoped\n                # schedule bag used while the sidecar is built/run.\n                _side_server_args = _mtp_copy.copy(server_args)\n                object.__setattr__(_side_server_args, "page_size", 1)\n                if getattr(server_args, "kv_cache_dtype", None) == "nvfp4":\n                    object.__setattr__(\n                        _side_server_args, "kv_cache_dtype", "bfloat16"\n                    )\n'''
PAGE_ARGS_MARKER = 'object.__setattr__(_side_server_args, "page_size", 1)'

POOL_OLD = '''                mr = self.draft_runner\n                logger.info(\n                    "[MTP-CUTOVER-KV] CUDA%d draft_kv_dtype=%s draft_kv_tag=%s",\n'''
POOL_NEW = '''                mr = self.draft_runner\n                if int(getattr(mr, "page_size", -1)) != 1:\n                    raise RuntimeError(\n                        "CUDA2 MTP sidecar must use page_size=1; got "\n                        f"{getattr(mr, 'page_size', None)} with "\n                        f"allocator={type(mr.token_to_kv_pool_allocator).__name__}"\n                    )\n                logger.info(\n                    "[MTP-CUTOVER-PAGE] CUDA%d draft_page_size=%d allocator=%s",\n                    self.gpu_id,\n                    int(mr.page_size),\n                    type(mr.token_to_kv_pool_allocator).__name__,\n                )\n                logger.info(\n                    "[MTP-CUTOVER-KV] CUDA%d draft_kv_dtype=%s draft_kv_tag=%s",\n'''
PAGE_POOL_MARKER = '"[MTP-CUTOVER-PAGE] CUDA%d draft_page_size=%d allocator=%s"'


def replace_once(
    text: str, old: str, new: str, label: str, path: pathlib.Path
) -> tuple[str, bool]:
    if new in text:
        return text, False
    if old not in text:
        raise RuntimeError(f"{label} patch point not found: {path}")
    return text.replace(old, new, 1), True


def patch_args(text: str, path: pathlib.Path) -> tuple[str, bool]:
    # Composition-safe invariant: once page_size=1 is explicitly installed on
    # the private sidecar args, this hotfix is complete regardless of whether
    # another hotfix selected FP8 or BF16 draft KV.
    if PAGE_ARGS_MARKER in text:
        return text, False
    if ARGS_OLD_BF16 in text:
        return text.replace(ARGS_OLD_BF16, ARGS_NEW_BF16, 1), True
    if ARGS_OLD_FP8 in text:
        return text.replace(ARGS_OLD_FP8, ARGS_NEW_FP8, 1), True
    raise RuntimeError(f"private sidecar args patch point not found: {path}")


def patch(path: pathlib.Path) -> bool:
    text = path.read_text()
    changed = False

    text, c = replace_once(text, IMPORT_OLD, IMPORT_NEW, "runtime-context import", path)
    changed |= c
    text, c = replace_once(text, CTX_OLD, CTX_NEW, "sidecar schedule context", path)
    changed |= c
    text, c = patch_args(text, path)
    changed |= c

    if PAGE_POOL_MARKER not in text:
        if POOL_OLD not in text:
            raise RuntimeError(f"sidecar page invariant patch point not found: {path}")
        text = text.replace(POOL_OLD, POOL_NEW, 1)
        changed = True

    if changed:
        path.write_text(text)
        subprocess.run(["python3", "-m", "py_compile", str(path)], check=True)
        print(f"fixed CUDA2 sidecar page_size=1 isolation: {path}")
    else:
        print(f"CUDA2 sidecar page_size=1 isolation already installed: {path}")
    return changed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--commit", action="store_true")
    args = ap.parse_args()

    host_changed = patch(HOST)
    fork_changed = patch(FORK)
    print(
        "MTP CUTOVER SIDECAR PAGE1 HOTFIX OK "
        f"host_changed={host_changed} fork_changed={fork_changed}"
    )

    if args.commit and fork_changed:
        subprocess.run(
            ["git", "add", str(FORK.relative_to(REPO))], cwd=REPO, check=True
        )
        diff = subprocess.run(
            ["git", "diff", "--cached", "--quiet"], cwd=REPO, check=False
        if diff.returncode != 0:
            subprocess.run(
                [
                    "git",
                    "commit",
                    "-m",
                    "fix: isolate CUDA2 MTP sidecar page size",
                ],
                cwd=REPO,
                check=True,
            )
            subprocess.run(
                ["git", "push", "origin", "HEAD:wip/qwen38-mtp-sidecar-cuda2"],
                cwd=REPO,
                check=True,
            )
            print("MTP CUTOVER SIDECAR PAGE1 COMMIT/PUSH OK")


if __name__ == "__main__":
    main()
