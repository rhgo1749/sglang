#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pathlib
import subprocess

HOME = pathlib.Path.home()
REPO = HOME / "projects/sglang-fork"
HOST = HOME / "projects/sglang-patches/eagle_worker_v2.sidecar-pool-probe.py"
FORK = REPO / "python/sglang/srt/speculative/eagle_worker_v2.py"

OLD = '''                with _mtp_sidecar_parallel_context(get_self_pp_group()):\n                    self._draft_worker = EagleDraftWorker(\n                        server_args,\n                        sidecar_gpu_id,\n                        sidecar_ps,\n                        nccl_port,\n                        target_worker,\n                        mtp_authoritative_sidecar=True,\n                    )\n'''

NEW = '''                # NVFP4 target decode is served by TRTLLM MHA, while the CUDA2\n                # EAGLE draft keeps the already-proven FlashInfer multi-step path.\n                # The exact image predates --speculative-draft-kv-cache-dtype, so\n                # give only the in-process sidecar a private ServerArgs view with\n                # FP8 E4M3 KV. Never mutate the published target ServerArgs.\n                import copy as _mtp_copy\n\n                _side_server_args = server_args\n                if getattr(server_args, "kv_cache_dtype", None) == "nvfp4":\n                    _side_server_args = _mtp_copy.copy(server_args)\n                    object.__setattr__(\n                        _side_server_args, "kv_cache_dtype", "fp8_e4m3"\n                    )\n                    logger.info(\n                        "[MTP-CUTOVER-KV] target=nvfp4 CUDA%d draft=fp8_e4m3",\n                        sidecar_gpu_id,\n                    )\n\n                with _mtp_sidecar_parallel_context(get_self_pp_group()):\n                    self._draft_worker = EagleDraftWorker(\n                        _side_server_args,\n                        sidecar_gpu_id,\n                        sidecar_ps,\n                        nccl_port,\n                        target_worker,\n                        mtp_authoritative_sidecar=True,\n                    )\n'''

LOG_OLD = '''                mr = self.draft_runner\n                self.req_to_token_pool = mr.req_to_token_pool\n'''
LOG_NEW = '''                mr = self.draft_runner\n                logger.info(\n                    "[MTP-CUTOVER-KV] CUDA%d draft_kv_dtype=%s draft_kv_tag=%s",\n                    self.gpu_id,\n                    str(getattr(mr, "kv_cache_dtype", None)),\n                    str(getattr(mr, "kv_cache_dtype_str", None)),\n                )\n                self.req_to_token_pool = mr.req_to_token_pool\n'''

CONSTRUCTION_MARKER = '"[MTP-CUTOVER-KV] target=nvfp4 CUDA%d draft=fp8_e4m3"'
LOG_MARKER = '"[MTP-CUTOVER-KV] CUDA%d draft_kv_dtype=%s draft_kv_tag=%s"'


def patch(path: pathlib.Path) -> bool:
    text = path.read_text()
    changed = False

    # Later page-size isolation intentionally rewrites the private ServerArgs
    # stanza while preserving this marker.  Treat that composed form as already
    # installed instead of requiring the byte-exact pre-page1 template.
    if CONSTRUCTION_MARKER not in text:
        if OLD not in text:
            raise RuntimeError(f"NVFP4 sidecar construction patch point not found: {path}")
        text = text.replace(OLD, NEW, 1)
        changed = True

    # Same for the page-size invariant inserted between `mr = ...` and this log.
    if LOG_MARKER not in text:
        if LOG_OLD not in text:
            raise RuntimeError(f"NVFP4 sidecar KV log patch point not found: {path}")
        text = text.replace(LOG_OLD, LOG_NEW, 1)
        changed = True

    if changed:
        path.write_text(text)
        subprocess.run(["python3", "-m", "py_compile", str(path)], check=True)
        print(f"fixed NVFP4-target / FP8-draft split: {path}")
    else:
        print(f"NVFP4-target / FP8-draft split already installed: {path}")
    return changed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--commit", action="store_true")
    args = ap.parse_args()

    host_changed = patch(HOST)
    fork_changed = patch(FORK)
    print(
        "MTP CUTOVER NVFP4 TARGET HOTFIX OK "
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
                [
                    "git",
                    "commit",
                    "-m",
                    "feat: split NVFP4 target KV from FP8 CUDA2 draft KV",
                ],
                cwd=REPO,
                check=True,
            )
            subprocess.run(
                ["git", "push", "origin", "HEAD:wip/qwen38-mtp-sidecar-cuda2"],
                cwd=REPO,
                check=True,
            )
            print("MTP CUTOVER NVFP4 TARGET COMMIT/PUSH OK")


if __name__ == "__main__":
    main()
