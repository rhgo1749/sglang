#!/usr/bin/env python3
"""Apply the local Qwen3.8 CUDA2 MTP sidecar fixes in one shot.

This branch stores exact-image copies of eagle_worker_v2.py and qwen3_5_mtp.py
from lmsysorg/sglang:qwen38-27b.  The surrounding fork main can move ahead of
that image, so this patcher deliberately edits those exact-image copies in
~/projects/sglang-patches first, then syncs them into this WIP branch.

Root causes fixed here:
  * draft_tp_context patches _TP but not _ATTN_TP;
  * Qwen3.5/3.8 attention layers read get_parallel().attn_tp_* at construction;
  * the CUDA2 worker was therefore TP1 for generic layers but still TP2 for
    attention projections + LayerCommunicator;
  * swapping only the runtime NCCL group cannot repair those static TP2 shapes;
  * SGLang RuntimeContext resources are process-global.  FlashInfer explicitly
    reuses get_buffer("flashinfer_workspace", ...), so an in-process CUDA2
    sidecar otherwise aliases the target CUDA0 FlashInfer workspace.  The
    sidecar forward can finish and still corrupt the authoritative target's
    next FlashInfer plan.

The sidecar-specific context below therefore makes TP, attention-TP,
attention-CP and attention-DP consistently singleton AND gives the sidecar its
own named persistent-buffer / named-stream registries.  Both overlays are
stacked and restored transactionally around construction, pool/backend init,
ForwardBatch creation and the actual forward.
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

    helper = '''\n\n_MTP_SIDECAR_BUFFERS = {}\n_MTP_SIDECAR_STREAMS = {}\n\n\n@contextlib.contextmanager\ndef _mtp_sidecar_parallel_context(tp_group):\n    """Temporarily make the CUDA2 sidecar a coherent isolated TP1 runtime.\n\n    SGLang's normal draft_tp_context only swaps _TP.  Qwen3.5/3.8 attention\n    construction and LayerCommunicator also consume the separate attention\n    topology, so a TP1 sidecar built with the target's _ATTN_TP becomes a\n    permanently half-sharded attention model.\n\n    RuntimeContext named buffers/streams are process-global as well.  In\n    particular FlashInfer reuses the key ``flashinfer_workspace`` across all\n    wrappers.  A second model on another CUDA device must not inherit the\n    target GPU's raw workspace pointer.  Keep a persistent private registry for\n    the sidecar and restore the target registry immediately on scope exit.\n    """\n    import sglang.srt.distributed.parallel_state as _ps\n    from sglang.srt.layers import dp_attention as _dp\n\n    _saved_tp = _ps._TP\n    _saved_attn_tp = _ps._ATTN_TP\n    _saved_attn_cp = _ps._ATTN_CP\n    _saved_attn_dp_size = _dp._ATTN_DP_SIZE\n    _saved_attn_dp_rank = _dp._ATTN_DP_RANK\n\n    _rank = int(getattr(tp_group, "rank_in_group", 0))\n    _size = int(getattr(tp_group, "world_size", 1))\n    if _size != 1:\n        raise RuntimeError(\n            f"MTP sidecar requires a singleton TP group, got world_size={_size}"\n        )\n\n    _ps._TP = tp_group\n    _ps._ATTN_TP = tp_group\n    _ps._ATTN_CP = tp_group\n    _dp._ATTN_DP_SIZE = 1\n    _dp._ATTN_DP_RANK = 0\n\n    try:\n        with (\n            get_context().resources.override(\n                buffers=_MTP_SIDECAR_BUFFERS,\n                streams=_MTP_SIDECAR_STREAMS,\n            ),\n            get_parallel().override(\n                tp_size=1,\n                tp_rank=_rank,\n                tp_group=tp_group,\n                attn_tp_size=1,\n                attn_tp_rank=_rank,\n                attn_tp_group=tp_group,\n                attn_cp_size=1,\n                attn_cp_rank=0,\n                attn_cp_group=tp_group,\n                attn_dp_size=1,\n                attn_dp_rank=0,\n                dcp_enabled=False,\n                dcp_size=1,\n                dcp_rank=0,\n                attn_dcp_size=1,\n                attn_dcp_rank=0,\n            ),\n        ):\n            yield\n    finally:\n        _dp._ATTN_DP_RANK = _saved_attn_dp_rank\n        _dp._ATTN_DP_SIZE = _saved_attn_dp_size\n        _ps._ATTN_CP = _saved_attn_cp\n        _ps._ATTN_TP = _saved_attn_tp\n        _ps._TP = _saved_tp\n'''

    if "def _mtp_sidecar_parallel_context(" not in s:
        marker = "logger = logging.getLogger(__name__)\n"
        if marker not in s:
            raise RuntimeError("eagle logger insertion point not found")
        s = s.replace(marker, marker + helper, 1)
    else:
        # Upgrade an already topology-patched file to process-resource isolation.
        if "_MTP_SIDECAR_BUFFERS = {}" not in s:
            marker = "logger = logging.getLogger(__name__)\n"
            if marker not in s:
                raise RuntimeError("eagle logger insertion point not found")
            s = s.replace(
                marker,
                marker + "\n\n_MTP_SIDECAR_BUFFERS = {}\n_MTP_SIDECAR_STREAMS = {}\n",
                1,
            )

        if "get_context().resources.override(" not in s:
            old = '''    try:\n        with get_parallel().override(\n            tp_size=1,\n            tp_rank=_rank,\n            tp_group=tp_group,\n            attn_tp_size=1,\n            attn_tp_rank=_rank,\n            attn_tp_group=tp_group,\n            attn_cp_size=1,\n            attn_cp_rank=0,\n            attn_cp_group=tp_group,\n            attn_dp_size=1,\n            attn_dp_rank=0,\n            dcp_enabled=False,\n            dcp_size=1,\n            dcp_rank=0,\n            attn_dcp_size=1,\n            attn_dcp_rank=0,\n        ):\n            yield\n'''
            new = '''    try:\n        with (\n            get_context().resources.override(\n                buffers=_MTP_SIDECAR_BUFFERS,\n                streams=_MTP_SIDECAR_STREAMS,\n            ),\n            get_parallel().override(\n                tp_size=1,\n                tp_rank=_rank,\n                tp_group=tp_group,\n                attn_tp_size=1,\n                attn_tp_rank=_rank,\n                attn_tp_group=tp_group,\n                attn_cp_size=1,\n                attn_cp_rank=0,\n                attn_cp_group=tp_group,\n                attn_dp_size=1,\n                attn_dp_rank=0,\n                dcp_enabled=False,\n                dcp_size=1,\n                dcp_rank=0,\n                attn_dcp_size=1,\n                attn_dcp_rank=0,\n            ),\n        ):\n            yield\n'''
            if old not in s:
                raise RuntimeError("eagle sidecar context upgrade point not found")
            s = s.replace(old, new, 1)

    # Every get_self_pp_group() TP context in this WIP file belongs to the
    # CUDA2 sidecar probe.  Replace it with the coherent topology/resource overlay.
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

    # Validate the exact process-global collision that caused the post-success
    # target FlashInfer illegal access.  After leaving the sidecar context, the
    # target registry must point at its CUDA0 workspace while the sidecar keeps
    # a distinct CUDA2 workspace in its private registry.
    if "[MTP-SIDECAR-RESOURCE]" not in s:
        marker = '''                mr = self._mtp_sidecar_probe.model_runner\n                logger.info(\n                    "[MTP-SIDECAR-ATTN] CUDA%d attention backend ready: %s",\n'''
        if marker not in s:
            raise RuntimeError("eagle sidecar attention resource insertion point not found")
        replacement = '''                mr = self._mtp_sidecar_probe.model_runner\n\n                _side_ws = _MTP_SIDECAR_BUFFERS.get("flashinfer_workspace")\n                _target_ws = get_context().resources.buffers.get("flashinfer_workspace")\n                if isinstance(_side_ws, torch.Tensor):\n                    if _side_ws.device != torch.device("cuda", sidecar_gpu_id):\n                        raise RuntimeError(\n                            "CUDA2 sidecar FlashInfer workspace landed on the wrong device: "\n                            f"{_side_ws.device}"\n                        )\n                    if isinstance(_target_ws, torch.Tensor) and (\n                        _side_ws.data_ptr() == _target_ws.data_ptr()\n                    ):\n                        raise RuntimeError(\n                            "CUDA2 sidecar still aliases the target FlashInfer workspace"\n                        )\n\n                logger.info(\n                    "[MTP-SIDECAR-RESOURCE] side_flashinfer=%s target_flashinfer=%s separate=%s",\n                    (str(_side_ws.device) if isinstance(_side_ws, torch.Tensor) else None),\n                    (str(_target_ws.device) if isinstance(_target_ws, torch.Tensor) else None),\n                    (\n                        not isinstance(_side_ws, torch.Tensor)\n                        or not isinstance(_target_ws, torch.Tensor)\n                        or _side_ws.data_ptr() != _target_ws.data_ptr()\n                    ),\n                )\n                logger.info(\n                    "[MTP-SIDECAR-ATTN] CUDA%d attention backend ready: %s",\n'''
        s = s.replace(marker, replacement, 1)

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
        backup = src.with_suffix(src.suffix + ".before-resource-isolation")
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
                "wip: isolate CUDA2 MTP sidecar runtime resources",
                cwd=repo,
            )
        run("git", "push", "origin", f"HEAD:{BRANCH}", cwd=repo)
        print("COMMIT/PUSH OK")


if __name__ == "__main__":
    main()
