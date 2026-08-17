#!/usr/bin/env python3
import ast
from pathlib import Path

QPATH = Path(
    "/sgl-workspace/sglang/python/sglang/srt/layers/quantization/fp4_kv_cache_quant_method.py"
)
MPATH = Path("/sgl-workspace/sglang/python/sglang/srt/mem_cache/memory_pool.py")

qtext = QPATH.read_text()
mtext = MPATH.read_text()

# Keep the original dequantize_prev_kv fallback memory-safe as well.  This is
# still used by any path that does not opt into the direct-to-workspace helper.
seq_old = '''        k_bf16 = NVFP4KVQuantizeUtil.dequantize(\n            k_fp4.view(torch.uint8), k_scales, cur_k_scale\n        )\n        v_bf16 = NVFP4KVQuantizeUtil.dequantize(\n            v_fp4.view(torch.uint8), v_scales, cur_v_scale\n        )\n        return k_bf16.to(torch.float8_e4m3fn), v_bf16.to(torch.float8_e4m3fn)\n'''
seq_new = '''        # Keep peak temporary memory bounded during very long prefix dequant.\n        # Convert K immediately and release its BF16 temporary before\n        # materializing V.  Returned tensors and numerical formats are\n        # unchanged; only temporary lifetimes are shortened.\n        k_bf16 = NVFP4KVQuantizeUtil.dequantize(\n            k_fp4.view(torch.uint8), k_scales, cur_k_scale\n        )\n        k_fp8 = k_bf16.to(torch.float8_e4m3fn)\n        del k_bf16\n\n        v_bf16 = NVFP4KVQuantizeUtil.dequantize(\n            v_fp4.view(torch.uint8), v_scales, cur_v_scale\n        )\n        v_fp8 = v_bf16.to(torch.float8_e4m3fn)\n        del v_bf16\n\n        return k_fp8, v_fp8\n'''
if seq_old in qtext:
    qtext = qtext.replace(seq_old, seq_new, 1)
elif seq_new not in qtext:
    raise SystemExit("ERROR: NVFP4 fallback dequant source shape changed; refusing blind patch")

# IMPORTANT: insert the helper inside NVFP4KVCacheMethod, not before the first
# compute_cell_size() in the file.  The latter sits under @abstractmethod in the
# base class; inserting there accidentally makes the helper abstract.
quant_marker = "def dequantize_prev_kv_into_workspace("
quant_helper = '''    def dequantize_prev_kv_into_workspace(\n        self,\n        k_fp4: Tensor,\n        k_scales: Tensor,\n        v_fp4: Tensor,\n        v_scales: Tensor,\n        src_indices: Tensor,\n        layer_id: int,\n        dst_k: Tensor,\n        dst_v: Tensor,\n        dst_indices: Optional[Tensor] = None,\n    ) -> None:\n        \"\"\"Stream indexed FP4 KV directly into the shared FP8 workspace.\n\n        Only one packed gather and one BF16 dequant temporary are live at a\n        time.  Long prefixes are processed in bounded chunks so the temporary\n        footprint does not grow linearly all the way to 256K tokens.  Assignment\n        into the preallocated FP8 destination performs the BF16->FP8 cast\n        without materializing a full-size FP8 result tensor.\n        \"\"\"\n        from sglang.srt.layers.quantization.kvfp4_tensor import NVFP4KVQuantizeUtil\n\n        cur_k_scale = self.k_scales_gpu[layer_id : layer_id + 1]\n        cur_v_scale = self.v_scales_gpu[layer_id : layer_id + 1]\n        dequant_chunk_tokens = 65536\n        total = int(src_indices.numel())\n\n        for start in range(0, total, dequant_chunk_tokens):\n            end = min(start + dequant_chunk_tokens, total)\n            src_chunk = src_indices[start:end]\n            k_fp4_sel = k_fp4[src_chunk]\n            k_scales_sel = k_scales[src_chunk]\n            k_bf16 = NVFP4KVQuantizeUtil.dequantize(\n                k_fp4_sel.view(torch.uint8), k_scales_sel, cur_k_scale\n            )\n            if dst_indices is None:\n                dst_k[start:end].copy_(k_bf16)\n            else:\n                dst_k[dst_indices[start:end]] = k_bf16\n            del k_bf16, k_fp4_sel, k_scales_sel, src_chunk\n\n        for start in range(0, total, dequant_chunk_tokens):\n            end = min(start + dequant_chunk_tokens, total)\n            src_chunk = src_indices[start:end]\n            v_fp4_sel = v_fp4[src_chunk]\n            v_scales_sel = v_scales[src_chunk]\n            v_bf16 = NVFP4KVQuantizeUtil.dequantize(\n                v_fp4_sel.view(torch.uint8), v_scales_sel, cur_v_scale\n            )\n            if dst_indices is None:\n                dst_v[start:end].copy_(v_bf16)\n            else:\n                dst_v[dst_indices[start:end]] = v_bf16\n            del v_bf16, v_fp4_sel, v_scales_sel, src_chunk\n\n'''

nv_start = qtext.find("class NVFP4KVCacheMethod(KVCacheQuantMethodBase):")
nv_end = qtext.find("\n\nclass FP4MXBlock16KVCacheMethod", nv_start)
if nv_start < 0 or nv_end < 0:
    raise SystemExit("ERROR: could not locate NVFP4KVCacheMethod class boundaries")

nv_block = qtext[nv_start:nv_end]
if quant_marker not in nv_block:
    compute_pos = nv_block.find("    def compute_cell_size(\n")
    if compute_pos < 0:
        raise SystemExit("ERROR: NVFP4 compute_cell_size anchor not found")
    nv_block = nv_block[:compute_pos] + quant_helper + nv_block[compute_pos:]
    qtext = qtext[:nv_start] + nv_block + qtext[nv_end:]

extend_marker = "# MTP-PP-NVFP4 direct extend dequant into shared workspace"
extend_old = '''            if prev_len > 0:\n                prev_indices = req_to_token[req_idx, :prev_len]\n                k_prev_fp8, v_prev_fp8 = self.quant_method.dequantize_prev_kv(\n                    k_fp4[prev_indices],\n                    k_scales[prev_indices],\n                    v_fp4[prev_indices],\n                    v_scales[prev_indices],\n                    global_layer_id,\n                )\n                dq_k[cur_token_idx_dq : cur_token_idx_dq + prev_len] = k_prev_fp8\n                dq_v[cur_token_idx_dq : cur_token_idx_dq + prev_len] = v_prev_fp8\n'''
extend_new = '''            if prev_len > 0:\n                prev_indices = req_to_token[req_idx, :prev_len]\n                dst_start = cur_token_idx_dq\n                dst_end = cur_token_idx_dq + prev_len\n                if hasattr(self.quant_method, \"dequantize_prev_kv_into_workspace\"):\n                    # MTP-PP-NVFP4 direct extend dequant into shared workspace\n                    self.quant_method.dequantize_prev_kv_into_workspace(\n                        k_fp4,\n                        k_scales,\n                        v_fp4,\n                        v_scales,\n                        prev_indices,\n                        global_layer_id,\n                        dq_k[dst_start:dst_end],\n                        dq_v[dst_start:dst_end],\n                    )\n                else:\n                    k_prev_fp8, v_prev_fp8 = self.quant_method.dequantize_prev_kv(\n                        k_fp4[prev_indices],\n                        k_scales[prev_indices],\n                        v_fp4[prev_indices],\n                        v_scales[prev_indices],\n                        global_layer_id,\n                    )\n                    dq_k[dst_start:dst_end] = k_prev_fp8\n                    dq_v[dst_start:dst_end] = v_prev_fp8\n'''
if extend_marker not in mtext:
    if extend_old not in mtext:
        raise SystemExit("ERROR: NVFP4 extend workspace source shape changed; refusing blind patch")
    mtext = mtext.replace(extend_old, extend_new, 1)

decode_marker = "# MTP-PP-NVFP4 direct decode dequant into shared workspace"
decode_old = '''            kv_indices = req_to_token[req_idx, :seq_len]\n            k_prev_fp8, v_prev_fp8 = self.quant_method.dequantize_prev_kv(\n                k_fp4[kv_indices],\n                k_scales[kv_indices],\n                v_fp4[kv_indices],\n                v_scales[kv_indices],\n                global_layer_id,\n            )\n            dq_k[kv_indices] = k_prev_fp8\n            dq_v[kv_indices] = v_prev_fp8\n'''
decode_new = '''            kv_indices = req_to_token[req_idx, :seq_len]\n            if hasattr(self.quant_method, \"dequantize_prev_kv_into_workspace\"):\n                # MTP-PP-NVFP4 direct decode dequant into shared workspace\n                self.quant_method.dequantize_prev_kv_into_workspace(\n                    k_fp4,\n                    k_scales,\n                    v_fp4,\n                    v_scales,\n                    kv_indices,\n                    global_layer_id,\n                    dq_k,\n                    dq_v,\n                    dst_indices=kv_indices,\n                )\n            else:\n                k_prev_fp8, v_prev_fp8 = self.quant_method.dequantize_prev_kv(\n                    k_fp4[kv_indices],\n                    k_scales[kv_indices],\n                    v_fp4[kv_indices],\n                    v_scales[kv_indices],\n                    global_layer_id,\n                )\n                dq_k[kv_indices] = k_prev_fp8\n                dq_v[kv_indices] = v_prev_fp8\n'''
if decode_marker not in mtext:
    if decode_old not in mtext:
        raise SystemExit("ERROR: NVFP4 decode workspace source shape changed; refusing blind patch")
    mtext = mtext.replace(decode_old, decode_new, 1)

# Syntax + structural guards.  In particular, never let the new helper become
# an abstract method on KVCacheQuantMethodBase again.
qtree = ast.parse(qtext, filename=str(QPATH))
ast.parse(mtext, filename=str(MPATH))
classes = {
    node.name: node for node in qtree.body if isinstance(node, ast.ClassDef)
}
base = classes.get("KVCacheQuantMethodBase")
nv = classes.get("NVFP4KVCacheMethod")
if base is None or nv is None:
    raise SystemExit("ERROR: expected KV quant classes not found after patch")
base_methods = {
    node.name for node in base.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
}
nv_methods = {
    node.name: node for node in nv.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
}
if "dequantize_prev_kv_into_workspace" in base_methods:
    raise SystemExit("ERROR: direct dequant helper was inserted into abstract base class")
helper = nv_methods.get("dequantize_prev_kv_into_workspace")
if helper is None:
    raise SystemExit("ERROR: direct dequant helper missing from NVFP4KVCacheMethod")
if any(
    isinstance(dec, ast.Name) and dec.id == "abstractmethod"
    for dec in helper.decorator_list
):
    raise SystemExit("ERROR: NVFP4 direct dequant helper unexpectedly abstract")

QPATH.write_text(qtext)
MPATH.write_text(mtext)
print(
    "[MTP-PP-NVFP4-DEQUANT] installed concrete-class direct-to-workspace "
    "sequential/chunked K/V dequant patch"
)
