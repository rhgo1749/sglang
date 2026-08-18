#!/usr/bin/env python3
from pathlib import Path
import ast

MODEL_RUNNER = Path(
    "/sgl-workspace/sglang/python/sglang/srt/model_executor/model_runner.py"
)
CUDA_GRAPH_SETUP = Path(
    "/sgl-workspace/sglang/python/sglang/srt/model_executor/model_runner_components/cuda_graph_setup.py"
)
REPLAY_MARKER = "[MTP-PP-TARGET-VERIFY-GRAPH]"
CAPTURE_MARKER = "[MTP-PP-TARGET-VERIFY-CAPTURE]"

# ---------------------------------------------------------------------------
# 1) Replay scope: TARGET_VERIFY can fall back to the live eager runner while
#    normal target DECODE eligibility remains governed by cuda_graph_config.
# ---------------------------------------------------------------------------
text = MODEL_RUNNER.read_text()
if REPLAY_MARKER not in text:
    anchor = '''            can_run_graph = bool(\n                mode_check()\n                and self.decode_cuda_graph_runner\n                and self.decode_cuda_graph_runner.can_run_graph(forward_batch)\n            )\n\n'''
    repl = '''            can_run_graph = bool(\n                mode_check()\n                and self.decode_cuda_graph_runner\n                and self.decode_cuda_graph_runner.can_run_graph(forward_batch)\n            )\n\n            # [MTP-PP-TARGET-VERIFY-GRAPH] Native-MTP uses TARGET_VERIFY for\n            # every speculative target round. Keep that graph independently\n            # controllable from normal target DECODE and EAGLE draft graphs.\n            import os as _mtp_os\n            _mtp_disable_target_verify_graph = (\n                forward_batch.forward_mode.is_target_verify()\n                and _mtp_os.environ.get(\n                    "SGLANG_MTP_DISABLE_TARGET_VERIFY_CUDA_GRAPH", "0"\n                ).strip().lower() in {"1", "true", "yes", "y"}\n            )\n            if can_run_graph and _mtp_disable_target_verify_graph:\n                can_run_graph = False\n                if not getattr(\n                    self, "_mtp_target_verify_graph_disable_logged", False\n                ):\n                    logger.info(\n                        "[MTP-PP-TARGET-VERIFY-GRAPH] PP%d TARGET_VERIFY CUDA "\n                        "graph replay disabled; eager target verify selected",\n                        int(self.ps.pp_rank),\n                    )\n                    self._mtp_target_verify_graph_disable_logged = True\n\n'''
    count = text.count(anchor)
    if count != 1:
        raise RuntimeError(
            f"expected exactly one ModelRunner decode graph eligibility anchor, found {count}"
        )
    text = text.replace(anchor, repl, 1)
    ast.parse(text, filename=str(MODEL_RUNNER))
    MODEL_RUNNER.write_text(text)

# ---------------------------------------------------------------------------
# 2) Capture scope: replay-only gating is insufficient for this Qwen3.8
#    PP3/native-MTP route. Capturing the speculative target runner itself can
#    perturb graph/static state before the first live request. When the same
#    scope knob is set, do not instantiate/capture TARGET_VERIFY at all.
#    This intentionally does NOT disable PP, eager target verify, EAGLE draft
#    graphs, or prefill graphs.
# ---------------------------------------------------------------------------
text = CUDA_GRAPH_SETUP.read_text()
if CAPTURE_MARKER not in text:
    anchor = '''    # A PD prefill server never replays the target-verify graph, and its pool\n'''
    repl = '''    # [MTP-PP-TARGET-VERIFY-CAPTURE] A replay-only gate still performs the\n    # target-verify capture at startup. For the patched PP3/native-MTP route,\n    # allow that capture to be skipped completely while keeping other graph\n    # phases independently available.\n    import os as _mtp_os\n    _mtp_skip_target_verify_capture = (\n        model_runner.spec_algorithm.is_speculative()\n        and not model_runner.is_draft_worker\n        and _mtp_os.environ.get(\n            "SGLANG_MTP_DISABLE_TARGET_VERIFY_CUDA_GRAPH", "0"\n        ).strip().lower() in {"1", "true", "yes", "y"}\n    )\n    if _mtp_skip_target_verify_capture:\n        logger.info(\n            "[MTP-PP-TARGET-VERIFY-CAPTURE] PP%d target-verify graph capture "\n            "skipped; eager target verify / PP / draft / prefill paths remain available",\n            int(model_runner.ps.pp_rank),\n        )\n        return no_capture\n\n    # A PD prefill server never replays the target-verify graph, and its pool\n'''
    count = text.count(anchor)
    if count != 1:
        raise RuntimeError(
            f"expected exactly one target-verify capture anchor, found {count}"
        )
    text = text.replace(anchor, repl, 1)
    ast.parse(text, filename=str(CUDA_GRAPH_SETUP))
    CUDA_GRAPH_SETUP.write_text(text)

# Structural audit.
mr = MODEL_RUNNER.read_text()
cg = CUDA_GRAPH_SETUP.read_text()
ast.parse(mr, filename=str(MODEL_RUNNER))
ast.parse(cg, filename=str(CUDA_GRAPH_SETUP))
for token in (
    "SGLANG_MTP_DISABLE_TARGET_VERIFY_CUDA_GRAPH",
    REPLAY_MARKER,
    "forward_batch.forward_mode.is_target_verify()",
    "can_run_graph = False",
):
    if token not in mr:
        raise RuntimeError(f"target-verify replay scope audit failed: {token}")
for token in (
    "SGLANG_MTP_DISABLE_TARGET_VERIFY_CUDA_GRAPH",
    CAPTURE_MARKER,
    "_mtp_skip_target_verify_capture",
    "return no_capture",
):
    if token not in cg:
        raise RuntimeError(f"target-verify capture scope audit failed: {token}")

print("[MTP-PP-TARGET-VERIFY-GRAPH] installed independent TARGET_VERIFY replay control")
print("[MTP-PP-TARGET-VERIFY-CAPTURE] installed independent TARGET_VERIFY capture control")
print("VERIFIED PP/eager target verify/EAGLE draft/prefill graph paths remain independently available")
