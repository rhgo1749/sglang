#!/usr/bin/env python3
"""Repair CUDA2 authoritative MTP sidecar Mamba tracking ownership.

The authoritative CUDA2 draft owns an independent HybridReqToTokenPool.  A
normal colocated EAGLE draft shares the target request pool, so the process-wide
`extra_buffer_lazy` Mamba tracking helpers can rely on the target pool's
ping-pong mapping.  The independent draft pool is created as a draft-worker pool
and may not allocate that mapping.  The first real prefill then reaches
`set_mamba_track_indices_from_reqs()` and crashes before the sidecar forward.

For this intentionally narrow cutover (max_running_requests=1,
extra_buffer_lazy), promote the independent sidecar pool to the same lazy
ping-pong tracking contract immediately after allocation.  HybridReqToTokenPool
already has the allocator and `_alloc_ping_pong_buffer()` implementation; once
`enable_mamba_extra_buffer` is true, its normal `alloc(reqs)` path owns slot
allocation and populates the mapping.

This hotfix is idempotent and repairs:
  * the host exact-image source mounted into Docker,
  * the fork checkpoint,
  * the cutover generator template.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

BRANCH = "wip/qwen38-mtp-sidecar-cuda2"
TAG = "[MTP-CUTOVER-MAMBA]"

RUNTIME_MARKER = '''                self.req_to_token_pool = mr.req_to_token_pool
                self.token_to_kv_pool_allocator = mr.token_to_kv_pool_allocator
'''

RUNTIME_INSERT = '''                self.req_to_token_pool = mr.req_to_token_pool
                self.token_to_kv_pool_allocator = mr.token_to_kv_pool_allocator

                # A colocated EAGLE draft shares the target HybridReqToTokenPool,
                # whose extra-buffer ping-pong mapping is created by the target
                # pool.  This CUDA2 draft owns an independent draft-worker pool;
                # exact-image builds can therefore omit that mapping even though
                # the process-wide Mamba strategy is extra_buffer_lazy.  Promote
                # the sidecar pool to the same lazy tracking contract before its
                # first Req allocation.  HybridReqToTokenPool.alloc() then owns
                # the actual ping-pong slot allocation and mapping updates.
                if (
                    hasattr(self.req_to_token_pool, "mamba_pool")
                    and not hasattr(
                        self.req_to_token_pool,
                        "req_index_to_mamba_ping_pong_track_buffer_mapping",
                    )
                ):
                    _side_pool = self.req_to_token_pool
                    _side_pool.enable_mamba_extra_buffer = True
                    _side_pool.enable_mamba_extra_buffer_lazy = True
                    if not hasattr(_side_pool, "mamba_ping_pong_track_buffer_size"):
                        _side_pool.mamba_ping_pong_track_buffer_size = 2
                    _side_pool.req_index_to_mamba_ping_pong_track_buffer_mapping = (
                        torch.zeros(
                            (
                                _side_pool.req_to_token.shape[0],
                                _side_pool.mamba_ping_pong_track_buffer_size,
                            ),
                            dtype=torch.int64,
                            device=_side_pool.device,
                        )
                    )
                    logger.info(
                        "[MTP-CUTOVER-MAMBA] CUDA%d installed lazy ping-pong "
                        "tracking rows=%d width=%d mamba_slots=%d",
                        self.gpu_id,
                        _side_pool.req_to_token.shape[0],
                        _side_pool.mamba_ping_pong_track_buffer_size,
                        _side_pool.mamba_pool.size,
                    )
'''

# Same insertion, but represented inside qwen38_mtp_sidecar_cutover.py's
# triple-quoted generated source where newlines are written as literal `\\n`.
GEN_MARKER = RUNTIME_MARKER.replace("\n", "\\n")
GEN_INSERT = RUNTIME_INSERT.replace("\n", "\\n")


def run(*args: str, cwd: Path | None = None) -> None:
    subprocess.run(args, cwd=cwd, check=True)


def patch_runtime(path: Path) -> bool:
    if not path.exists():
        raise RuntimeError(f"required runtime source not found: {path}")
    text = path.read_text()
    if TAG in text:
        print(f"already fixed: {path}")
        return False
    if text.count(RUNTIME_MARKER) != 1:
        raise RuntimeError(
            f"expected one authoritative pool marker in {path}, "
            f"found {text.count(RUNTIME_MARKER)}"
        )
    path.write_text(text.replace(RUNTIME_MARKER, RUNTIME_INSERT, 1))
    print(f"fixed sidecar Mamba tracking: {path}")
    return True


def patch_generator(path: Path) -> bool:
    if not path.exists():
        raise RuntimeError(f"required generator not found: {path}")
    text = path.read_text()
    if TAG in text:
        print(f"generator already fixed: {path}")
        return False
    if text.count(GEN_MARKER) != 1:
        raise RuntimeError(
            f"expected one generator pool marker in {path}, "
            f"found {text.count(GEN_MARKER)}"
        )
    path.write_text(text.replace(GEN_MARKER, GEN_INSERT, 1))
    print(f"fixed generator Mamba tracking: {path}")
    return True


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    patch_dir = Path.home() / "projects" / "sglang-patches"
    host_eagle = patch_dir / "eagle_worker_v2.sidecar-pool-probe.py"
    repo_eagle = repo / "python/sglang/srt/speculative/eagle_worker_v2.py"
    generator = repo / "tools/qwen38_mtp_sidecar_cutover.py"

    changed_host = patch_runtime(host_eagle)
    repo_eagle.write_text(host_eagle.read_text())
    print(f"synced runtime source: {repo_eagle}")
    changed_generator = patch_generator(generator)

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
            "fix: initialize CUDA2 MTP Mamba tracking",
            cwd=repo,
        )
        run("git", "push", "origin", f"HEAD:{BRANCH}", cwd=repo)
        print("MTP CUTOVER MAMBA HOTFIX COMMIT/PUSH OK")
    else:
        print("MTP CUTOVER MAMBA HOTFIX ALREADY APPLIED")

    print(
        "MTP CUTOVER MAMBA HOTFIX OK "
        f"host_changed={changed_host} generator_changed={changed_generator}"
    )


if __name__ == "__main__":
    main()
