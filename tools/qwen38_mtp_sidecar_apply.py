#!/usr/bin/env python3
"""Apply the local Qwen3.8 CUDA2 MTP sidecar topology fix in one shot.

This branch stores exact-image copies of eagle_worker_v2.py and qwen3_5_mtp.py
from lmsysorg/sglang:qwen38-27b.  The surrounding fork main can move ahead of
that image, so this patcher deliberately edits those exact-image copies in
~/projects/sglang-patches first, then syncs them into this WIP branch.

Root cause fixed here:
  * draft_tp_context patches _TP but not _ATTN_TP;
  * Qwen3.5/3.8 attention layers read get_parallel().attn_tp_* at construction;
  * the CUDA2 worker was therefore TP1 for generic layers but still TP2 for
    attention projections + LayerCommunicator;
  * swapping only the runtime NCCL group cannot repair those static TP2 shapes;
  * the resulting communicator later dereferenced DP-padding metadata that a
    real TP1 sidecar does not need.

The sidecar-specific context below makes TP, attention-TP, attention-CP and
attention-DP consistently singleton for construction, pool/backend init,
ForwardBatch creation and the actual forward.  It is re-entrant by pointer
stacking and therefore does not fight SGLang's _TP_STATE_PATCHED guard.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
from pathlib import Path

BRANCH = "wip/qwen38-mtp-sidecar-cuda2"


def run(*args: str, cwd: Path | None = None) -> None:
    subprocess.run(args, cwd=cwd, check=True)


def patch_eagle(path: Path) -> None:
    s = path.read_text()

    helper = '''\n\n@contextlib.contextmanager\ndef _mtp_sidecar_parallel_context(tp_group):\n    """Temporarily make the CUDA2 sidecar a coherent TP1/ATTN-TP1 runtime.\n\n    SGLang's normal draft_tp_context only swaps _TP.  Qwen3.5/3.8 attention\n    construction and LayerCommunicator also consume the separate attention\n    topology, so a TP1 sidecar built with the target's _ATTN_TP becomes a\n    permanently half-sharded attention model.  Stack the relevant pointers\n    directly instead of nesting patch_tensor_parallel_group; this is safe both\n    inside and outside an existing draft context and restores every value.\n    """\n    import sglang.srt.distributed.parallel_state as _ps\n    from sglang.srt.layers import dp_attention as _dp\n\n    _saved_tp = _ps._TP\n    _saved_attn_tp = _ps._ATTN_TP\n    _saved_attn_cp = _ps._ATTN_CP\n    _saved_attn_dp_size = _dp._ATTN_DP_SIZE\n    _saved_attn_dp_rank = _dp._ATTN_DP_RANK\n\n    _rank = int(getattr(tp_group, "rank_in_group", 0))\n    _size = int(getattr(tp_group, "world_size", 1))\n    if _size != 1:\n        raise RuntimeError(\n            f"MTP sidecar requires a singleton TP group, got world_size={_size}"\n        )\n\n    _ps._TP = tp_group\n    _ps._ATTN_TP = tp_group\n    _ps._ATTN_CP = tp_group\n    _dp._ATTN_DP_SIZE = 1\n    _dp._ATTN_DP_RANK = 0\n\n    try:\n        with get_parallel().override(\n            tp_size=1,\n            tp_rank=_rank,\n            tp_group=tp_group,\n            attn_tp_size=1,\n            attn_tp_rank=_rank,\n            attn_tp_group=tp_group,\n            attn_cp_size=1,\n            attn_cp_rank=0,\n            attn_cp_group=tp_group,\n            attn_dp_size=1,\n            attn_dp_rank=0,\n            dcp_enabled=False,\n            dcp_size=1,\n            dcp_rank=0,\n            attn_dcp_size=1,\n            attn_dcp_rank=0,\n        ):\n            yield\n    finally:\n        _dp._ATTN_DP_RANK = _saved_attn_dp_rank\n        _dp._ATTN_DP_SIZE = _saved_attn_dp_size\n        _ps._ATTN_CP = _saved_attn_cp\n        _ps._ATTN_TP = _saved_attn_tp\n        _ps._TP = _saved_tp\n'''

    if "def _mtp_sidecar_parallel_context(" not in s:
        marker = "logger = logging.getLogger(__name__)\n"
        if marker not in s:
            raise RuntimeError("eagle logger insertion point not found")
        s = s.replace(marker, marker + helper, 1)

    # Every get_self_pp_group() TP context in this WIP file belongs to the
    # CUDA2 sidecar probe.  Replace it with the coherent topology overlay.
    s = s.replace(
        "draft_tp_context(get_self_pp_group()),",
        "_mtp_sidecar_parallel_context(get_self_pp_group()),",
    )

    # The shadow helper historically opened a second draft_tp_context inside
    # its already-active sidecar context.  Remove both old variants (nested
    # context and the later direct _TP/_ATTN_TP workaround).  The enclosing
    # sidecar context now owns the entire ForwardBatch + forward lifetime.
    start_marker = "                # The normal draft_tp_context only patches _TP."
    end_marker = "            if side_logits.next_token_logits is None:"
    start = s.find(start_marker)
    if start != -1:
        end = s.find(end_marker, start)
        if end == -1:
            raise RuntimeError("eagle sidecar forward block end not found")
        replacement = '''                # ForwardBatch.init_new and the model forward are already\n                # enclosed by _mtp_sidecar_parallel_context, so do not nest\n                # SGLang's non-reentrant draft TP/PP patchers here.\n                side_logits = mr.forward(forward_batch).logits_output\n\n                # Surface asynchronous CUDA faults on CUDA2 here.  Otherwise a\n                # shadow-side illegal access can be reported later by the\n                # authoritative target draft and produce a misleading traceback.\n                torch.cuda.synchronize(sidecar_gpu_id)\n\n'''
        s = s[:start] + replacement + s[end:]

    # Fail early if the model was ever constructed with target TP2 attention
    # metadata.  Runtime group swapping cannot repair QKV/O-proj shard shapes or
    # LayerCommunicator's captured CommunicateContext.
    topo_marker = '''            mr = sidecar.model_runner\n\n            # get_rope() uses a process-wide module cache.'''
    if "[MTP-SIDECAR-TOPO]" not in s:
        if topo_marker not in s:
            raise RuntimeError("eagle topology validation insertion point not found")
        topo_block = '''            mr = sidecar.model_runner\n\n            if not getattr(self, "_mtp_sidecar_topology_checked", False):\n                _attn_sizes = set()\n                _comm_attn_sizes = set()\n                _comm_tp_sizes = set()\n                for _name, _mod in mr.model.named_modules():\n                    _attn_size = getattr(_mod, "attn_tp_size", None)\n                    if _attn_size is not None:\n                        _attn_sizes.add(int(_attn_size))\n                    _comm = getattr(_mod, "layer_communicator", None)\n                    _ctx = getattr(_comm, "_context", None)\n                    if _ctx is not None:\n                        _comm_attn_sizes.add(int(_ctx.attn_tp_size))\n                        _comm_tp_sizes.add(int(_ctx.tp_size))\n\n                if _attn_sizes and _attn_sizes != {1}:\n                    raise RuntimeError(\n                        f"CUDA2 sidecar attention was not built TP1: {_attn_sizes}"\n                    )\n                if _comm_attn_sizes and _comm_attn_sizes != {1}:\n                    raise RuntimeError(\n                        "CUDA2 sidecar LayerCommunicator captured non-TP1 "\n                        f"attention topology: {_comm_attn_sizes}"\n                    )\n                if _comm_tp_sizes and _comm_tp_sizes != {1}:\n                    raise RuntimeError(\n                        "CUDA2 sidecar LayerCommunicator captured non-TP1 "\n                        f"generic topology: {_comm_tp_sizes}"\n                    )\n\n                self._mtp_sidecar_topology_checked = True\n                logger.info(\n                    "[MTP-SIDECAR-TOPO] static attention=%s communicator_attn=%s communicator_tp=%s",\n                    sorted(_attn_sizes),\n                    sorted(_comm_attn_sizes),\n                    sorted(_comm_tp_sizes),\n                )\n\n            # get_rope() uses a process-wide module cache.'''
        s = s.replace(topo_marker, topo_block, 1)

    path.write_text(s)


def patch_qwen(path: Path) -> None:
    s = path.read_text()
    marker = '''        self._lm_head_quant_config = (\n            original_quant_config\n            if self._load_full_embed_head\n            else quant_config\n        )\n\n        self.quant_config = quant_config\n'''
    if "TP1 MTP sidecar was constructed with non-TP1 attention topology" not in s:
        if marker not in s:
            raise RuntimeError("qwen sidecar topology assertion point not found")
        replacement = '''        self._lm_head_quant_config = (\n            original_quant_config\n            if self._load_full_embed_head\n            else quant_config\n        )\n\n        # The self-contained ModelOpt sidecar must be TP1 for BOTH generic TP\n        # and attention TP before Qwen3_5ForCausalLM is constructed.  QKV/O-proj\n        # shard sizes and LayerCommunicator topology are captured here and cannot\n        # be repaired by swapping an NCCL group later during forward.\n        if self._load_full_embed_head and (\n            self.tp_size != 1\n            or get_parallel().attn_tp_size != 1\n            or get_parallel().attn_dp_size != 1\n            or get_parallel().attn_cp_size != 1\n        ):\n            raise RuntimeError(\n                "TP1 MTP sidecar was constructed with non-TP1 attention topology: "\n                f"tp={self.tp_size}, attn_tp={get_parallel().attn_tp_size}, "\n                f"attn_dp={get_parallel().attn_dp_size}, "\n                f"attn_cp={get_parallel().attn_cp_size}"\n            )\n\n        self.quant_config = quant_config\n'''
        s = s.replace(marker, replacement, 1)
    path.write_text(s)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--commit", action="store_true", help="commit and push the WIP branch")
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[1]
    patch_dir = Path.home() / "projects" / "sglang-patches"
    eagle_src = patch_dir / "eagle_worker_v2.sidecar-pool-probe.py"
    qwen_src = patch_dir / "qwen3_5_mtp.sidecar.py"

    if not eagle_src.exists() or not qwen_src.exists():
        raise SystemExit(
            "expected exact-image patch files under ~/projects/sglang-patches"
        )

    for src in (eagle_src, qwen_src):
        backup = src.with_suffix(src.suffix + ".before-topology-fix")
        if not backup.exists():
            shutil.copy2(src, backup)

    patch_eagle(eagle_src)
    patch_qwen(qwen_src)

    eagle_dst = repo / "python/sglang/srt/speculative/eagle_worker_v2.py"
    qwen_dst = repo / "python/sglang/srt/models/qwen3_5_mtp.py"
    shutil.copy2(eagle_src, eagle_dst)
    shutil.copy2(qwen_src, qwen_dst)

    run("python3", "-m", "py_compile", str(eagle_src), str(qwen_src))
    run("git", "diff", "--check", cwd=repo)

    print("PATCHED OK")
    print(f"  eagle: {eagle_src}")
    print(f"  qwen : {qwen_src}")

    if args.commit:
        run(
            "git",
            "add",
            "python/sglang/srt/speculative/eagle_worker_v2.py",
            "python/sglang/srt/models/qwen3_5_mtp.py",
            cwd=repo,
        )
        status = subprocess.run(
            ["git", "diff", "--cached", "--quiet"], cwd=repo
        )
        if status.returncode != 0:
            run(
                "git",
                "commit",
                "-m",
                "wip: fix CUDA2 MTP sidecar TP1 attention topology",
                cwd=repo,
            )
        run("git", "push", "origin", f"HEAD:{BRANCH}", cwd=repo)
        print("COMMIT/PUSH OK")


if __name__ == "__main__":
    main()
