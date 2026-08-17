#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pathlib
import subprocess

HOME = pathlib.Path.home()
REPO = HOME / "projects/sglang-fork"
HOST = HOME / "projects/sglang-patches/eagle_worker_v2.sidecar-pool-probe.py"
FORK = REPO / "python/sglang/srt/speculative/eagle_worker_v2.py"

HYBRID_MARKER = '"[MTP-CUTOVER-KV] target=nvfp4 CUDA%d draft=fp8_e4m3"'
HIGHWATER_MARKER = '"[MTP-CUTOVER-HIGHWATER] CUDA%d page_size=%d allocator=%s"'


def _replace_or_keep(text: str, old: str, new: str, label: str, path: pathlib.Path):
    if new in text:
        return text, False
    if old not in text:
        raise RuntimeError(f"{label} patch point not found: {path}")
    return text.replace(old, new, 1), True


def _patch_private_args(text: str, path: pathlib.Path):
    start_marker = "                _side_server_args = _mtp_copy.copy(server_args)\n"
    end_marker = "                with _mtp_sidecar_parallel_context(get_self_pp_group()):\n"
    start = text.find(start_marker)
    if start < 0:
        raise RuntimeError(f"private sidecar args start not found: {path}")
    end = text.find(end_marker, start)
    if end < 0:
        raise RuntimeError(f"private sidecar args end not found: {path}")

    block = text[start:end]
    old_block = block

    block = block.replace(
        'object.__setattr__(_side_server_args, "page_size", 64)',
        'object.__setattr__(_side_server_args, "page_size", 1)',
    )
    block = block.replace(
        '                        _side_server_args, "kv_cache_dtype", "nvfp4"',
        '                        _side_server_args, "kv_cache_dtype", "fp8_e4m3"',
    )
    block = block.replace(
        '                        _side_server_args, "speculative_draft_attention_backend", None',
        '                        _side_server_args, "speculative_draft_attention_backend", "flashinfer"',
    )
    block = block.replace(
        '                        _side_server_args, "decode_attention_backend", "trtllm_mha"',
        '                        _side_server_args, "decode_attention_backend", "flashinfer"',
    )
    block = block.replace(
        '"[MTP-CUTOVER-KV] target=nvfp4 CUDA%d draft=nvfp4"',
        '"[MTP-CUTOVER-KV] target=nvfp4 CUDA%d draft=fp8_e4m3"',
    )
    block = block.replace(
        "# Match target and CUDA2 draft KV quantization for the NVFP4\n"
        "                # acceptance A/B. NVFP4 supports FlashInfer for prefill/extend\n"
        "                # and TRTLLM-MHA for decode, so remove the single draft-backend\n"
        "                # override and let the private split backends resolve per phase.\n",
        "# Production hybrid after the NVFP4-draft acceptance A/B: keep the\n"
        "                # capacity-critical target KV in NVFP4, but put the latency-critical\n"
        "                # CUDA2 EAGLE draft back on the proven FP8 E4M3 + FlashInfer path.\n",
    )

    required = (
        '"page_size", 1',
        '"kv_cache_dtype", "fp8_e4m3"',
        '"speculative_draft_attention_backend", "flashinfer"',
        '"prefill_attention_backend", "flashinfer"',
        '"decode_attention_backend", "flashinfer"',
        HYBRID_MARKER.strip("'"),
    )
    missing = [x for x in required if x not in block]
    if missing:
        raise RuntimeError(f"hybrid private args incomplete in {path}: {missing}")

    return text[:start] + block + text[end:], block != old_block


def patch(path: pathlib.Path) -> bool:
    text = path.read_text()
    changed = False

    text, c = _replace_or_keep(
        text,
        "get_schedule().override(page_size=64),",
        "get_schedule().override(page_size=1),",
        "sidecar RuntimeContext page size",
        path,
    )
    changed |= c

    text, c = _patch_private_args(text, path)
    changed |= c

    text, c = _replace_or_keep(
        text,
        'if int(getattr(mr, "page_size", -1)) != 64:',
        'if int(getattr(mr, "page_size", -1)) != 1:',
        "sidecar page assertion",
        path,
    )
    changed |= c
    if '"CUDA2 MTP sidecar must use page_size=64; got "' in text:
        text = text.replace(
            '"CUDA2 MTP sidecar must use page_size=64; got "',
            '"CUDA2 MTP sidecar must use page_size=1; got "',
            1,
        )
        changed = True

    if '"[MTP-CUTOVER-PAGED] CUDA%d page_size=%d high-water allocator enabled"' in text:
        text = text.replace(
            '"[MTP-CUTOVER-PAGED] CUDA%d page_size=%d high-water allocator enabled"',
            '"[MTP-CUTOVER-HIGHWATER] CUDA%d page_size=%d allocator=%s"',
            1,
        )
        old_args = '''                    self.gpu_id,\n                    int(mr.page_size),\n                )'''
        new_args = '''                    self.gpu_id,\n                    int(mr.page_size),\n                    type(mr.token_to_kv_pool_allocator).__name__,\n                )'''
        marker_pos = text.find('[MTP-CUTOVER-HIGHWATER]')
        arg_pos = text.find(old_args, marker_pos)
        if arg_pos < 0:
            raise RuntimeError(f"high-water log arguments not found: {path}")
        text = text[:arg_pos] + text[arg_pos:].replace(old_args, new_args, 1)
        changed = True

    text = text.replace(
        "keep the sidecar on a TRTLLM-compatible paged allocator (page_size=64).",
        "keep the sidecar on the FlashInfer token allocator (page_size=1).",
        1,
    )
    text = text.replace(
        "Arbitrary speculative spans are handled by the high-water alloc_extend path below.",
        "Speculative spans still use the high-water helper so rejected tails are reused safely.",
        1,
    )

    for needle in (
        HYBRID_MARKER.strip("'"),
        'get_schedule().override(page_size=1)',
        'object.__setattr__(_side_server_args, "page_size", 1)',
        '"kv_cache_dtype", "fp8_e4m3"',
        '"speculative_draft_attention_backend", "flashinfer"',
        '"decode_attention_backend", "flashinfer"',
        "self._mtp_side_alloc_lens = {}",
        "def _mtp_authoritative_reserve_span(",
    ):
        if needle not in text:
            raise RuntimeError(f"hybrid postcondition missing {needle!r}: {path}")

    path.write_text(text)
    subprocess.run(["python3", "-m", "py_compile", str(path)], check=True)
    if changed:
        print(f"installed NVFP4-target / FP8-FlashInfer CUDA2 hybrid: {path}")
    else:
        print(f"hybrid already installed: {path}")
    return changed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--commit", action="store_true")
    args = ap.parse_args()

    host_changed = patch(HOST)
    fork_changed = patch(FORK)
    print(
        "MTP CUTOVER HYBRID FP8 DRAFT HOTFIX OK "
        f"host_changed={host_changed} fork_changed={fork_changed}"
    )

    if args.commit and fork_changed:
        subprocess.run(["git", "add", str(FORK.relative_to(REPO))], cwd=REPO, check=True)
        diff = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=REPO, check=False)
        if diff.returncode != 0:
            subprocess.run(
                [
                    "git",
                    "commit",
                    "-m",
                    "fix: restore FP8 FlashInfer CUDA2 draft for NVFP4 target",
                ],
                cwd=REPO,
                check=True,
            )
            subprocess.run(
                ["git", "push", "origin", "HEAD:wip/qwen38-mtp-sidecar-cuda2"],
                cwd=REPO,
                check=True,
            )
            print("MTP CUTOVER HYBRID FP8 DRAFT COMMIT/PUSH OK")


if __name__ == "__main__":
    main()
