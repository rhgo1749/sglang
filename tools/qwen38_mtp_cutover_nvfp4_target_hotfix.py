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

NEW = '''                # Match target and CUDA2 draft KV quantization for the NVFP4\n                # acceptance A/B. NVFP4 supports FlashInfer for prefill/extend\n                # and TRTLLM-MHA for decode, so remove the single draft-backend\n                # override and let the private split backends resolve per phase.\n                # Never mutate the published target ServerArgs.\n                import copy as _mtp_copy\n\n                _side_server_args = server_args\n                if getattr(server_args, "kv_cache_dtype", None) == "nvfp4":\n                    _side_server_args = _mtp_copy.copy(server_args)\n                    object.__setattr__(\n                        _side_server_args, "kv_cache_dtype", "nvfp4"\n                    )\n                    object.__setattr__(\n                        _side_server_args, "speculative_draft_attention_backend", None\n                    )\n                    object.__setattr__(\n                        _side_server_args, "prefill_attention_backend", "flashinfer"\n                    )\n                    object.__setattr__(\n                        _side_server_args, "decode_attention_backend", "trtllm_mha"\n                    )\n                    logger.info(\n                        "[MTP-CUTOVER-KV] target=nvfp4 CUDA%d draft=nvfp4",\n                        sidecar_gpu_id,\n                    )\n\n                with _mtp_sidecar_parallel_context(get_self_pp_group()):\n                    self._draft_worker = EagleDraftWorker(\n                        _side_server_args,\n                        sidecar_gpu_id,\n                        sidecar_ps,\n                        nccl_port,\n                        target_worker,\n                        mtp_authoritative_sidecar=True,\n                    )\n'''

LOG_OLD = '''                mr = self.draft_runner\n                self.req_to_token_pool = mr.req_to_token_pool\n'''
LOG_NEW = '''                mr = self.draft_runner\n                logger.info(\n                    "[MTP-CUTOVER-KV] CUDA%d draft_kv_dtype=%s draft_kv_tag=%s",\n                    self.gpu_id,\n                    str(getattr(mr, "kv_cache_dtype", None)),\n                    str(getattr(mr, "kv_cache_dtype_str", None)),\n                )\n                self.req_to_token_pool = mr.req_to_token_pool\n'''

NVFP4_MARKER = '"[MTP-CUTOVER-KV] target=nvfp4 CUDA%d draft=nvfp4"'
BF16_MARKER = '"[MTP-CUTOVER-KV] target=nvfp4 CUDA%d draft=bfloat16"'
FP8_MARKER = '"[MTP-CUTOVER-KV] target=nvfp4 CUDA%d draft=fp8_e4m3"'
LOG_MARKER = '"[MTP-CUTOVER-KV] CUDA%d draft_kv_dtype=%s draft_kv_tag=%s"'
SPLIT_MARKER = '"speculative_draft_attention_backend", None'


def migrate_existing_split(text: str) -> tuple[str, bool]:
    """Migrate the prior FP8/BF16 sidecar experiment to NVFP4 + split backends."""
    if NVFP4_MARKER in text and SPLIT_MARKER in text:
        return text, False

    old_marker = BF16_MARKER if BF16_MARKER in text else FP8_MARKER if FP8_MARKER in text else None
    if old_marker is None:
        return text, False

    old_dtype = "bfloat16" if old_marker == BF16_MARKER else "fp8_e4m3"
    text = text.replace(
        f'object.__setattr__(\n                        _side_server_args, "kv_cache_dtype", "{old_dtype}"\n                    )',
        'object.__setattr__(\n                        _side_server_args, "kv_cache_dtype", "nvfp4"\n                    )\n'
        '                    object.__setattr__(\n'
        '                        _side_server_args, "speculative_draft_attention_backend", None\n'
        '                    )\n'
        '                    object.__setattr__(\n'
        '                        _side_server_args, "prefill_attention_backend", "flashinfer"\n'
        '                    )\n'
        '                    object.__setattr__(\n'
        '                        _side_server_args, "decode_attention_backend", "trtllm_mha"\n'
        '                    )',
        1,
    )
    text = text.replace(
        old_marker.strip("'"),
        NVFP4_MARKER.strip("'"),
        1,
    )
    text = text.replace(
        "# BF16 KV. Never mutate the published target ServerArgs.",
        "# NVFP4 KV with split FlashInfer/TRTLLM-MHA backends. Never mutate the published target ServerArgs.",
        1,
    )
    text = text.replace(
        "# FP8 E4M3 KV. Never mutate the published target ServerArgs.",
        "# NVFP4 KV with split FlashInfer/TRTLLM-MHA backends. Never mutate the published target ServerArgs.",
        1,
    )
    return text, True


def patch(path: pathlib.Path) -> bool:
    text = path.read_text()
    changed = False

    text, migrated = migrate_existing_split(text)
    changed |= migrated

    if NVFP4_MARKER not in text:
        if OLD not in text:
            raise RuntimeError(f"NVFP4 sidecar construction patch point not found: {path}")
        text = text.replace(OLD, NEW, 1)
        changed = True

    # A composed page-size hotfix may already have rewritten the private args.
    # The split marker is the invariant needed for NVFP4 draft decode.
    if SPLIT_MARKER not in text:
        raise RuntimeError(
            f"NVFP4 draft split-backend marker missing after patch: {path}"
        )

    if LOG_MARKER not in text:
        if LOG_OLD not in text:
            raise RuntimeError(f"NVFP4 sidecar KV log patch point not found: {path}")
        text = text.replace(LOG_OLD, LOG_NEW, 1)
        changed = True

    if changed:
        path.write_text(text)
        subprocess.run(["python3", "-m", "py_compile", str(path)], check=True)
        print(f"fixed NVFP4-target / NVFP4-draft split backend: {path}")
    else:
        print(f"NVFP4-target / NVFP4-draft split already installed: {path}")
    return changed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--commit", action="store_true")
    args = ap.parse_args()

    host_changed = patch(HOST)
    fork_changed = patch(FORK)
    print(
        "MTP CUTOVER NVFP4 TARGET+DRAFT HOTFIX OK "
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
                    "test: align CUDA2 MTP draft KV with NVFP4 target",
                ],
                cwd=REPO,
                check=True,
            )
            subprocess.run(
                ["git", "push", "origin", "HEAD:wip/qwen38-mtp-sidecar-cuda2"],
                cwd=REPO,
                check=True,
            )
            print("MTP CUTOVER NVFP4 TARGET+DRAFT COMMIT/PUSH OK")


if __name__ == "__main__":
    main()
