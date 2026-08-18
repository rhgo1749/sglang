from pathlib import Path
import ast

MTP = Path("/sgl-workspace/sglang/python/sglang/srt/models/qwen3_5_mtp.py")
s = MTP.read_text()

MARKER = "[MTP-PP-INPUT-RANGE-CG-SAFE]"

old = '''                if input_ids.numel() > 0:\n                    _min_id = int(input_ids.min().item())\n                    _max_id = int(input_ids.max().item())\n                else:\n                    _min_id = 0\n                    _max_id = -1\n                if _min_id < 0 or _max_id >= _rows:\n                    raise RuntimeError(\n                        "[MTP-PP-INPUT-OOB] "\n                        f"input_ids=[{_min_id},{_max_id}] "\n                        f"embed_rows={_rows} "\n                        f"config_vocab={self.config.vocab_size} "\n                        f"mode={forward_batch.forward_mode}"\n                    )\n                logger.info(\n                    "[MTP-PP-INPUT-RANGE] input_ids=[%d,%d] "\n                    "embed_rows=%d config_vocab=%d",\n                    _min_id, _max_id, _rows, int(self.config.vocab_size),\n                )\n'''

new = '''                # [MTP-PP-INPUT-RANGE-CG-SAFE] The synchronous .item() range\n                # preflight is useful in eager mode, but CUDA Graph capture\n                # forbids the device->host synchronization it causes. Capture\n                # uses preallocated/static ids, so keep the same embedding path\n                # and omit only this diagnostic host sync while capturing.\n                _capturing = torch.cuda.is_current_stream_capturing()\n                if not _capturing:\n                    if input_ids.numel() > 0:\n                        _min_id = int(input_ids.min().item())\n                        _max_id = int(input_ids.max().item())\n                    else:\n                        _min_id = 0\n                        _max_id = -1\n                    if _min_id < 0 or _max_id >= _rows:\n                        raise RuntimeError(\n                            "[MTP-PP-INPUT-OOB] "\n                            f"input_ids=[{_min_id},{_max_id}] "\n                            f"embed_rows={_rows} "\n                            f"config_vocab={self.config.vocab_size} "\n                            f"mode={forward_batch.forward_mode}"\n                        )\n                    logger.info(\n                        "[MTP-PP-INPUT-RANGE] input_ids=[%d,%d] "\n                        "embed_rows=%d config_vocab=%d",\n                        _min_id, _max_id, _rows, int(self.config.vocab_size),\n                    )\n                else:\n                    logger.debug(\n                        "[MTP-PP-INPUT-RANGE-CG-SAFE] skipping host-sync range "\n                        "diagnostic during CUDA Graph capture; embed_rows=%d",\n                        _rows,\n                    )\n'''

if MARKER in s:
    print("native MTP CUDA-graph-safe input guard already present")
else:
    if old not in s:
        raise RuntimeError("MTP input-range preflight block not found")
    s = s.replace(old, new, 1)
    ast.parse(s, filename=str(MTP))
    MTP.write_text(s)
    print("PATCHED native MTP input preflight for CUDA Graph capture")

text = MTP.read_text()
if MARKER not in text:
    raise RuntimeError("CUDA-graph-safe MTP input guard marker missing")
if "torch.cuda.is_current_stream_capturing()" not in text:
    raise RuntimeError("CUDA capture-state guard missing")
ast.parse(text, filename=str(MTP))
print(MTP)
