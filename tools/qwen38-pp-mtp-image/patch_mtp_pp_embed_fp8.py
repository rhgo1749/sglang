from pathlib import Path
import ast

MTP = Path("/sgl-workspace/sglang/python/sglang/srt/models/qwen3_5_mtp.py")
EAGLE = Path("/sgl-workspace/sglang/python/sglang/srt/speculative/eagle_worker_v2.py")

METHOD_MARKER = "[MTP-PP-FP8-EMBED-METHOD]"
RUNTIME_MARKER = "[MTP-PP-FP8-EMBED]"

# ---------------------------------------------------------------------------
# 1) Add a TP1-only row-wise FP8 embedding representation to the native MTP
#    sidecar.  The serialized checkpoint loads the MTP body in BF16 and also
#    keeps a full 248320 x 5120 BF16 token table on PP-last.  The table is used
#    only for token lookup, so store each row in E4M3 plus one BF16 row scale and
#    reconstruct only the selected rows back to BF16.
#
#    Quantize in chunks: a whole-table BF16->FP32 temporary would itself be
#    several GiB and can OOM the 16 GiB PP-last GPU during boot.
# ---------------------------------------------------------------------------
s = MTP.read_text()
if METHOD_MARKER not in s:
    insert_before = "    def get_embed_and_head(self):\n"
    if insert_before not in s:
        raise RuntimeError("Qwen3.5 MTP get_embed_and_head insertion point not found")

    methods = '''    # [MTP-PP-FP8-EMBED-METHOD]\n    @torch.no_grad()\n    def compress_embed_fp8_rowwise(self, chunk_rows: int = 2048):\n        if self.tp_size != 1:\n            raise RuntimeError(\n                "MTP row-wise FP8 embedding compression is TP1-only; "\n                f"got tp_size={self.tp_size}"\n            )\n\n        layer = self.model.embed_tokens\n        existing_scale = getattr(layer, "_mtp_fp8_row_scale", None)\n        if existing_scale is not None:\n            return {\n                "old_bytes": int(layer.weight.numel() * 2),\n                "new_bytes": int(\n                    layer.weight.numel() * layer.weight.element_size()\n                    + existing_scale.numel() * existing_scale.element_size()\n                ),\n                "rows": int(layer.weight.shape[0]),\n                "hidden": int(layer.weight.shape[1]),\n            }\n\n        weight = layer.weight\n        if weight.device.type != "cuda":\n            raise RuntimeError(\n                "MTP row-wise FP8 embedding compression requires CUDA weight; "\n                f"got {weight.device}"\n            )\n        if weight.ndim != 2:\n            raise RuntimeError(\n                f"MTP embedding must be rank-2, got shape={tuple(weight.shape)}"\n            )\n        if weight.dtype not in (torch.bfloat16, torch.float16, torch.float32):\n            raise RuntimeError(\n                "Unexpected MTP embedding dtype before FP8 compression: "\n                f"{weight.dtype}"\n            )\n\n        fp8_dtype = torch.float8_e4m3fn\n        fp8_max = float(torch.finfo(fp8_dtype).max)\n        rows, hidden = map(int, weight.shape)\n        old_bytes = int(weight.numel() * weight.element_size())\n\n        # Allocate the final compact representation first, then use only a small\n        # FP32 working chunk.  This keeps peak conversion overhead near the final\n        # FP8 table size instead of materializing a multi-GiB FP32 clone.\n        qweight = torch.empty(\n            (rows, hidden), device=weight.device, dtype=fp8_dtype\n        )\n        row_scale = torch.empty(\n            (rows,), device=weight.device, dtype=torch.bfloat16\n        )\n\n        chunk_rows = max(1, int(chunk_rows))\n        for start in range(0, rows, chunk_rows):\n            end = min(start + chunk_rows, rows)\n            chunk = weight[start:end].float()\n            max_abs = chunk.abs().amax(dim=1)\n            # Zero rows (padding, if any) use scale=1 and remain exactly zero.\n            scale = torch.where(\n                max_abs > 0,\n                max_abs / fp8_max,\n                torch.ones_like(max_abs),\n            )\n            quant = (chunk / scale.unsqueeze(1)).clamp(\n                min=-fp8_max, max=fp8_max\n            ).to(fp8_dtype)\n            qweight[start:end].copy_(quant)\n            row_scale[start:end].copy_(scale.to(torch.bfloat16))\n            del chunk, max_abs, scale, quant\n\n        # Weight loading and endpoint sharing have already completed before this\n        # method is called, so the loader metadata on the old Parameter is no\n        # longer needed.  Keep the conventional `weight` parameter name for\n        # diagnostics and module introspection.\n        layer.weight = nn.Parameter(qweight, requires_grad=False)\n        layer.register_buffer(\n            "_mtp_fp8_row_scale", row_scale, persistent=False\n        )\n        del weight, qweight, row_scale\n        torch.cuda.empty_cache()\n        torch.cuda.synchronize()\n\n        scale_buf = layer._mtp_fp8_row_scale\n        new_bytes = int(\n            layer.weight.numel() * layer.weight.element_size()\n            + scale_buf.numel() * scale_buf.element_size()\n        )\n        return {\n            "old_bytes": old_bytes,\n            "new_bytes": new_bytes,\n            "rows": rows,\n            "hidden": hidden,\n        }\n\n    def _mtp_embed_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:\n        layer = self.model.embed_tokens\n        row_scale = getattr(layer, "_mtp_fp8_row_scale", None)\n        if row_scale is None:\n            return layer(input_ids)\n\n        # TP1 sidecar: token ids map directly to full-vocab rows.  Gather the\n        # tiny set of requested FP8 rows, then dequantize only those rows.\n        original_shape = tuple(input_ids.shape)\n        flat_ids = input_ids.long().reshape(-1)\n        rows = torch.index_select(layer.weight, 0, flat_ids).to(torch.bfloat16)\n        scales = torch.index_select(row_scale, 0, flat_ids).to(torch.bfloat16)\n        rows = rows * scales.unsqueeze(1)\n        return rows.reshape(*original_shape, int(layer.embedding_dim))\n\n'''
    s = s.replace(insert_before, methods + insert_before, 1)

    # Restrict call-site rewrites to Qwen3_5ForCausalLMMTP.forward only.  The
    # underlying one-layer model continues to own a normal embedding module for
    # metadata/introspection, but every actual MTP token lookup goes through the
    # row-wise dequant helper.
    fstart = s.find("    @torch.no_grad()\n    def forward(\n")
    if fstart < 0:
        raise RuntimeError("Qwen3.5 MTP forward method not found")
    fend = s.find("\n    def load_weights(\n", fstart)
    if fend < 0:
        raise RuntimeError("Qwen3.5 MTP load_weights boundary not found")
    forward_block = s[fstart:fend]
    call_count = forward_block.count("self.model.embed_tokens(")
    if call_count != 2:
        raise RuntimeError(\n            "Expected exactly two MTP embedding lookup call sites in forward; "\n            f"found {call_count}"\n        )
    forward_block = forward_block.replace(\n        "self.model.embed_tokens(", "self._mtp_embed_tokens(", 2\n    )
    s = s[:fstart] + forward_block + s[fend:]
    ast.parse(s, filename=str(MTP))
    MTP.write_text(s)

# ---------------------------------------------------------------------------
# 2) Compress only after init_token_map()/init_lm_head() have finished endpoint
#    sharing, and before KV pool profiling.  This lets PP2's newly-freed memory
#    feed directly into its local token-capacity calculation.
# ---------------------------------------------------------------------------
s = EAGLE.read_text()
if RUNTIME_MARKER not in s:
    audit_marker = (
        "        # Diagnostic only: account physical CUDA storages after endpoint sharing.\n"
    )
    if audit_marker not in s:
        raise RuntimeError("MTP early-share memory-audit insertion point not found")

    runtime = '''        _mtp_embed_model = self.draft_runner.model\n        if not hasattr(_mtp_embed_model, "compress_embed_fp8_rowwise"):\n            raise RuntimeError(\n                "native Qwen3.5 MTP model is missing FP8 embedding compressor"\n            )\n        _mtp_embed_free_before, _ = torch.cuda.mem_get_info()\n        _mtp_embed_stats = _mtp_embed_model.compress_embed_fp8_rowwise()\n        _mtp_embed_free_after, _ = torch.cuda.mem_get_info()\n        logger.info(\n            "[MTP-PP-FP8-EMBED] PP%d old=%.3fGiB new=%.3fGiB "\n            "logical_saved=%.3fGiB free_delta=%.3fGiB rows=%d hidden=%d",\n            int(get_pp_group().rank_in_group),\n            _mtp_embed_stats["old_bytes"] / (1 << 30),\n            _mtp_embed_stats["new_bytes"] / (1 << 30),\n            (_mtp_embed_stats["old_bytes"] - _mtp_embed_stats["new_bytes"])\n            / (1 << 30),\n            (_mtp_embed_free_after - _mtp_embed_free_before) / (1 << 30),\n            int(_mtp_embed_stats["rows"]),\n            int(_mtp_embed_stats["hidden"]),\n        )\n\n'''
    s = s.replace(audit_marker, runtime + audit_marker, 1)
    ast.parse(s, filename=str(EAGLE))
    EAGLE.write_text(s)

# ---------------------------------------------------------------------------
# Build-time semantic audit.
# ---------------------------------------------------------------------------
mtp_text = MTP.read_text()
eagle_text = EAGLE.read_text()
for required in (
    METHOD_MARKER,
    "def compress_embed_fp8_rowwise",
    "def _mtp_embed_tokens",
    "torch.float8_e4m3fn",
    "_mtp_fp8_row_scale",
):
    if required not in mtp_text:
        raise RuntimeError(f"MTP FP8 embedding patch missing: {required}")

fstart = mtp_text.find("    @torch.no_grad()\n    def forward(\n")
fend = mtp_text.find("\n    def load_weights(\n", fstart)
if fstart < 0 or fend < 0:
    raise RuntimeError("MTP forward audit boundaries missing")
forward_block = mtp_text[fstart:fend]
if "self.model.embed_tokens(" in forward_block:
    raise RuntimeError("raw BF16 embedding lookup remains in MTP forward")
if forward_block.count("self._mtp_embed_tokens(") != 2:
    raise RuntimeError("MTP forward must route exactly two lookups through FP8 helper")

if eagle_text.count(RUNTIME_MARKER) != 1:
    raise RuntimeError(\n        "MTP FP8 embedding runtime hook must be injected exactly once; "\n        f"count={eagle_text.count(RUNTIME_MARKER)}"\n    )
if eagle_text.find(RUNTIME_MARKER) > eagle_text.find("[MTP-MEM-AUDIT]"):
    raise RuntimeError("MTP FP8 embedding compression must run before memory audit")
if eagle_text.find(RUNTIME_MARKER) < eagle_text.find("self.init_lm_head()"):
    raise RuntimeError("MTP FP8 embedding compression must run after endpoint sharing")

ast.parse(mtp_text, filename=str(MTP))
ast.parse(eagle_text, filename=str(EAGLE))

print("PATCHED PP-last native MTP row-wise FP8 embedding")
print("VERIFIED chunked FP8 conversion after endpoint sharing and before KV sizing")
print(MTP)
print(EAGLE)
