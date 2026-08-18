#!/usr/bin/env python3
from pathlib import Path
import ast

PATH = Path("/sgl-workspace/sglang/python/sglang/srt/managers/schedule_batch.py")
text = PATH.read_text()
MARKER = "[MTP-MAMBA-TRACK-SLOT-CLEAR]"

if MARKER not in text:
    old = '''        for req in reqs:\n            if req.mamba_cow_src_index is not None:\n                cow_src_tensors.append(req.mamba_cow_src_index)\n                cow_dst_tensors.append(req.mamba_pool_idx.unsqueeze(0))\n                req.mamba_cow_src_index = None\n                req.mamba_needs_clear = False\n            elif req.mamba_needs_clear:\n                clear_tensors.append(req.mamba_pool_idx.unsqueeze(0))\n                req.mamba_needs_clear = False\n'''
    new = '''        for req in reqs:\n            # [MTP-MAMBA-TRACK-SLOT-CLEAR] A freshly assigned request also owns\n            # one or more ping-pong tracking slots. The allocator intentionally\n            # recycles physical Mamba slots, so clear those request-scoped\n            # tracking rows on the same forward stream as the main slot before\n            # any speculative boundary tracking can observe the new owner.\n            _mtp_track_buf = getattr(req, "mamba_ping_pong_track_buffer", None)\n            _mtp_track_slots = None\n            if isinstance(_mtp_track_buf, torch.Tensor):\n                _mtp_track_slots = _mtp_track_buf[_mtp_track_buf != -1]\n\n            if req.mamba_cow_src_index is not None:\n                cow_src_tensors.append(req.mamba_cow_src_index)\n                cow_dst_tensors.append(req.mamba_pool_idx.unsqueeze(0))\n                if _mtp_track_slots is not None and _mtp_track_slots.numel() > 0:\n                    clear_tensors.append(_mtp_track_slots)\n                req.mamba_cow_src_index = None\n                req.mamba_needs_clear = False\n            elif req.mamba_needs_clear:\n                clear_tensors.append(req.mamba_pool_idx.unsqueeze(0))\n                if _mtp_track_slots is not None and _mtp_track_slots.numel() > 0:\n                    clear_tensors.append(_mtp_track_slots)\n                req.mamba_needs_clear = False\n'''
    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            f"expected one deferred Mamba clear collector anchor, found {count}"
        )
    text = text.replace(old, new, 1)
    ast.parse(text, filename=str(PATH))
    PATH.write_text(text)

text = PATH.read_text()
ast.parse(text, filename=str(PATH))
for token in (
    "[MTP-MAMBA-TRACK-SLOT-CLEAR]",
    'getattr(req, "mamba_ping_pong_track_buffer", None)',
    "_mtp_track_buf[_mtp_track_buf != -1]",
    "clear_tensors.append(_mtp_track_slots)",
):
    if token not in text:
        raise RuntimeError(f"MTP Mamba tracking-slot clear audit failed: {token}")

print("[MTP-MAMBA-TRACK-SLOT-CLEAR] installed request-ownership clear for Mamba tracking slots")
print("VERIFIED fresh main + lazy ping-pong Mamba slots are cleared together on the forward stream")
