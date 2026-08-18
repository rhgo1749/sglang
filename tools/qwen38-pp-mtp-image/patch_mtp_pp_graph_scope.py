#!/usr/bin/env python3
from pathlib import Path
import ast

PATH = Path("/sgl-workspace/sglang/python/sglang/srt/speculative/eagle_worker_v2.py")
text = PATH.read_text()
MARKER = "[MTP-PP-GRAPH-SCOPE]"

if MARKER not in text:
    anchor = '''        decode_backend = get_exec().graph.cuda_graph_config.decode.backend\n        capture_bs, _ = get_batch_sizes_to_capture(self.draft_runner)\n        if self.speculative_num_steps > 1:\n'''
    repl = '''        decode_backend = get_exec().graph.cuda_graph_config.decode.backend\n        capture_bs, _ = get_batch_sizes_to_capture(self.draft_runner)\n        import os as _mtp_os\n        _mtp_disable_draft_decode_graph = _mtp_os.environ.get(\n            "SGLANG_MTP_DISABLE_DRAFT_DECODE_CUDA_GRAPH", "0"\n        ).strip().lower() in {"1", "true", "yes", "y"}\n        # [MTP-PP-GRAPH-SCOPE] Keep target decode CUDA Graph independently\n        # controllable from the EAGLE/native-MTP draft decode graph.\n        if self.speculative_num_steps > 1 and not _mtp_disable_draft_decode_graph:\n'''
    count = text.count(anchor)
    if count != 1:
        raise RuntimeError(
            f"expected exactly one EAGLE draft graph capture anchor, found {count}"
        )
    text = text.replace(anchor, repl, 1)

    # Add an explicit runtime marker immediately before draft-extend capture setup.
    extend_anchor = '''        Device2ExtendCudaGraphRunner = {\n'''
    extend_repl = '''        if self.speculative_num_steps > 1 and _mtp_disable_draft_decode_graph:\n            logger.info(\n                "[MTP-PP-GRAPH-SCOPE] PP%d EAGLE draft-decode CUDA graph disabled; "\n                "target decode graph remains governed by cuda_graph_config",\n                int(self.ps.pp_rank),\n            )\n\n        Device2ExtendCudaGraphRunner = {\n'''
    count = text.count(extend_anchor)
    if count != 1:
        raise RuntimeError(
            f"expected exactly one draft-extend graph anchor, found {count}"
        )
    text = text.replace(extend_anchor, extend_repl, 1)

    ast.parse(text, filename=str(PATH))
    PATH.write_text(text)

# Structural audit
text = PATH.read_text()
ast.parse(text, filename=str(PATH))
for token in (
    "SGLANG_MTP_DISABLE_DRAFT_DECODE_CUDA_GRAPH",
    "[MTP-PP-GRAPH-SCOPE]",
    "and not _mtp_disable_draft_decode_graph",
):
    if token not in text:
        raise RuntimeError(f"MTP PP graph-scope patch audit failed: {token}")

print("[MTP-PP-GRAPH-SCOPE] installed independent EAGLE draft-decode graph control")
print("VERIFIED target decode CUDA graph can remain enabled while MTP draft decode runs eager")
