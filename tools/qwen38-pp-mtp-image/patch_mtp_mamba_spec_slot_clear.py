#!/usr/bin/env python3
from pathlib import Path
import ast

PATH = Path("/sgl-workspace/sglang/python/sglang/srt/mem_cache/memory_pool.py")
text = PATH.read_text()
MARKER = "[MTP-MAMBA-SPEC-SLOT-CLEAR]"

if MARKER not in text:
    method_anchor = '''    def clear_slots(self, indices: torch.Tensor):\n'''
    helper = '''    def _clear_speculative_slots(self, indices: torch.Tensor):\n        \"\"\"Clear request-scoped speculative scratch when a Mamba slot is reused.\"\"\"\n        if not isinstance(self.mamba_cache, self.SpeculativeState):\n            return\n\n        # [MTP-MAMBA-SPEC-SLOT-CLEAR] Native MTP target-verify stores rollback\n        # state in buffers keyed by the same physical Mamba slot as conv/temporal.\n        # A fresh request must not inherit those rows from the previous owner.\n        intermediate_ssm = self.mamba_cache.intermediate_ssm\n        if intermediate_ssm is not None and intermediate_ssm.numel() > 0:\n            intermediate_ssm[:, indices] = 0\n\n        # Clear the PHYSICAL backing tensors, not the overlapping as_strided\n        # logical views used by the deduplicated conv-window layout.\n        for phys in self._intermediate_conv_window_phys:\n            if phys.numel() > 0:\n                phys[:, indices] = 0\n\n'''
    count = text.count(method_anchor)
    if count != 1:
        raise RuntimeError(f"expected one MambaPool.clear_slots anchor, found {count}")
    text = text.replace(method_anchor, helper + method_anchor, 1)

    fused_anchor = '''            if temporal.numel() > 0:\n                temporal[:, indices] = 0\n            return\n'''
    fused_repl = '''            if temporal.numel() > 0:\n                temporal[:, indices] = 0\n            self._clear_speculative_slots(indices)\n            return\n'''
    count = text.count(fused_anchor)
    if count != 1:
        raise RuntimeError(f"expected one fused Mamba clear anchor, found {count}")
    text = text.replace(fused_anchor, fused_repl, 1)

    eager_anchor = '''            t = self.mamba_cache.temporal\n            t[:, indices] = 0\n\n    def copy_from(self, src_indices: torch.Tensor, dst_indices: torch.Tensor):\n'''
    eager_repl = '''            t = self.mamba_cache.temporal\n            t[:, indices] = 0\n\n        self._clear_speculative_slots(indices)\n\n    def copy_from(self, src_indices: torch.Tensor, dst_indices: torch.Tensor):\n'''
    count = text.count(eager_anchor)
    if count != 1:
        raise RuntimeError(f"expected one eager Mamba clear tail anchor, found {count}")
    text = text.replace(eager_anchor, eager_repl, 1)

    ast.parse(text, filename=str(PATH))
    PATH.write_text(text)

# Structural audit: both fused and non-fused clear paths must clear speculative
# scratch, while target-only State remains a no-op in the helper.
text = PATH.read_text()
ast.parse(text, filename=str(PATH))
for token in (
    "[MTP-MAMBA-SPEC-SLOT-CLEAR]",
    "def _clear_speculative_slots(self, indices: torch.Tensor):",
    "intermediate_ssm[:, indices] = 0",
    "for phys in self._intermediate_conv_window_phys:",
    "phys[:, indices] = 0",
):
    if token not in text:
        raise RuntimeError(f"MTP Mamba speculative slot-clear audit failed: {token}")
if text.count("self._clear_speculative_slots(indices)") != 2:
    raise RuntimeError("MTP Mamba speculative slot-clear must cover exactly two clear paths")

print("[MTP-MAMBA-SPEC-SLOT-CLEAR] installed speculative Mamba slot reuse clear")
print("VERIFIED native-MTP intermediate SSM and physical conv-window scratch are cleared on slot reuse")
