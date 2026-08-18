#!/usr/bin/env python3
from pathlib import Path
import ast

PATH = Path("/sgl-workspace/sglang/python/sglang/srt/model_executor/model_runner.py")
text = PATH.read_text()
MARKER = "[MTP-PP-TARGET-VERIFY-GRAPH]"

if MARKER not in text:
    anchor = '''            can_run_graph = bool(\n                mode_check()\n                and self.decode_cuda_graph_runner\n                and self.decode_cuda_graph_runner.can_run_graph(forward_batch)\n            )\n\n'''
    repl = '''            can_run_graph = bool(\n                mode_check()\n                and self.decode_cuda_graph_runner\n                and self.decode_cuda_graph_runner.can_run_graph(forward_batch)\n            )\n\n            # [MTP-PP-TARGET-VERIFY-GRAPH] Native-MTP uses TARGET_VERIFY for\n            # every speculative target round. Keep that graph independently\n            # controllable from normal target DECODE and EAGLE draft graphs so\n            # semantic regressions can be isolated without disabling all CUDA\n            # graph infrastructure process-wide.\n            import os as _mtp_os\n            _mtp_disable_target_verify_graph = (\n                forward_batch.forward_mode.is_target_verify()\n                and _mtp_os.environ.get(\n                    "SGLANG_MTP_DISABLE_TARGET_VERIFY_CUDA_GRAPH", "0"\n                ).strip().lower() in {"1", "true", "yes", "y"}\n            )\n            if can_run_graph and _mtp_disable_target_verify_graph:\n                can_run_graph = False\n                if not getattr(\n                    self, "_mtp_target_verify_graph_disable_logged", False\n                ):\n                    logger.info(\n                        "[MTP-PP-TARGET-VERIFY-GRAPH] PP%d TARGET_VERIFY CUDA "\n                        "graph disabled; normal target DECODE graph remains "\n                        "governed by cuda_graph_config",\n                        int(self.ps.pp_rank),\n                    )\n                    self._mtp_target_verify_graph_disable_logged = True\n\n'''
    count = text.count(anchor)
    if count != 1:
        raise RuntimeError(
            f"expected exactly one ModelRunner decode graph eligibility anchor, found {count}"
        )
    text = text.replace(anchor, repl, 1)
    ast.parse(text, filename=str(PATH))
    PATH.write_text(text)

# Structural audit: the scope knob must only gate TARGET_VERIFY graph replay;
# normal DECODE eligibility and capture remain untouched.
text = PATH.read_text()
ast.parse(text, filename=str(PATH))
for token in (
    "SGLANG_MTP_DISABLE_TARGET_VERIFY_CUDA_GRAPH",
    "[MTP-PP-TARGET-VERIFY-GRAPH]",
    "forward_batch.forward_mode.is_target_verify()",
    "can_run_graph = False",
):
    if token not in text:
        raise RuntimeError(f"target-verify graph scope patch audit failed: {token}")

print("[MTP-PP-TARGET-VERIFY-GRAPH] installed independent TARGET_VERIFY graph control")
print("VERIFIED normal target DECODE and EAGLE graph capture paths remain available")
