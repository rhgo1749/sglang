#!/usr/bin/env python3
from pathlib import Path

QPATH = Path(
    "/sgl-workspace/sglang/python/sglang/srt/layers/quantization/fp4_kv_cache_quant_method.py"
)
MPATH = Path("/sgl-workspace/sglang/python/sglang/srt/mem_cache/memory_pool.py")

qtext = QPATH.read_text()
mtext = MPATH.read_text()

quant_marker = "def dequantize_prev_kv_into_workspace("
quant_anchor = '''    def compute_cell_size(\n        self, head_num: int, head_dim: int, num_layers: int, kv_size: int\n    ) -> int:\n'''
quant_helper = '''    def dequantize_prev_kv_into_workspace(\n        self,\n        k_fp4: Tensor,\n        k_scales: Tensor,\n        v_fp4: Tensor,\n        v_scales: Tensor,\n        src_indices: Tensor,\n        layer_id: int,\n        dst_k: Tensor,\n        dst_v: Tensor,\n        dst_indices: Optional[Tensor] = None,\n    ) -> None:\n        \"\"\"Dequantize indexed FP4 KV directly into the shared FP8 workspace.\n\n        Keep only one packed gather and one BF16 dequant temporary live at a\n        time.  Assignment into the preallocated FP8 destination performs the\n        BF16->FP8 cast without materializing a second FP8 result tensor.\n        \"\"\"\n        from sglang.srt.layers.quantization.kvfp4_tensor import NVFP4KVQuantizeUtil\n\n        cur_k_scale = self.k_scales_gpu[layer_id : layer_id + 1]\n        cur_v_scale = self.v_scales_gpu[layer_id : layer_id + 1]\n\n        k_fp4_sel = k_fp4[src_indices]\n        k_scales_sel = k_scales[src_indices]\n        k_bf16 = NVFP4KVQuantizeUtil.dequantize(\n            k_fp4_sel.view(torch.uint8), k_scales_sel, cur_k_scale\n        )\n        if dst_indices is None:\n            dst_k.copy_(k_bf16)\n        else:\n            dst_k[dst_indices] = k_bf16\n        del k_bf16, k_fp4_sel, k_scales_sel\n\n        v_fp4_sel = v_fp4[src_indices]\n        v_scales_sel = v_scales[src_indices]\n        v_bf16 = NVFP4KVQuantizeUtil.dequantize(\n            v_fp4_sel.view(torch.uint8), v_scales_sel, cur_v_scale\n        )\n        if dst_indices is None:\n            dst_v.copy_(v_bf16)\n        else:\n            dst_v[dst_indices] = v_bf16\n        del v_bf16, v_fp4_sel, v_scales_sel\n\n'''

if quant_marker not in qtext:
    if quant_anchor not in qtext:
        raise SystemExit(
            "ERROR: NVFP4 quant method source shape changed; refusing blind patch"
        )
    qtext = qtext.replace(quant_anchor, quant_helper + quant_anchor, 1)

extend_marker = "# MTP-PP-NVFP4 direct extend dequant into shared workspace"
extend_old = '''            if prev_len > 0:\n                prev_indices = req_to_token[req_idx, :prev_len]\n                k_prev_fp8, v_prev_fp8 = self.quant_method.dequantize_prev_kv(\n                    k_fp4[prev_indices],\n                    k_scales[prev_indices],\n                    v_fp4[prev_indices],\n                    v_scales[prev_indices],\n                    global_layer_id,\n                )\n                dq_k[cur_token_idx_dq : cur_token_idx_dq + prev_len] = k_prev_fp8\n                dq_v[cur_token_idx_dq : cur_token_idx_dq + prev_len] = v_prev_fp8\n'''
extend_new = '''            if prev_len > 0:\n                prev_indices = req_to_token[req_idx, :prev_len]\n                dst_start = cur_token_idx_dq\n                dst_end = cur_token_idx_dq + prev_len\n                if hasattr(self.quant_method, \"dequantize_prev_kv_into_workspace\"):\n                    # MTP-PP-NVFP4 direct extend dequant into shared workspace\n                    self.quant_method.dequantize_prev_kv_into_workspace(\n                        k_fp4,\n                        k_scales,\n                        v_fp4,\n                        v_scales,\n                        prev_indices,\n                        global_layer_id,\n                        dq_k[dst_start:dst_end],\n                        dq_v[dst_start:dst_end],\n                    )\n                else:\n                    k_prev_fp8, v_prev_fp8 = self.quant_method.dequantize_prev_kv(\n                        k_fp4[prev_indices],\n                        k_scales[prev_indices],\n                        v_fp4[prev_indices],\n                        v_scales[prev_indices],\n                        global_layer_id,\n                    )\n                    dq_k[dst_start:dst_end] = k_prev_fp8\n                    dq_v[dst_start:dst_end] = v_prev_fp8\n'''

if extend_marker not in mtext:
    if extend_old not in mtext:
        raise SystemExit(
            "ERROR: NVFP4 extend workspace source shape changed; refusing blind patch"
        )
    mtext = mtext.replace(extend_old, extend_new, 1)

decode_marker = "# MTP-PP-NVFP4 direct decode dequant into shared workspace"
decode_old = '''            kv_indices = req_to_token[req_idx, :seq_len]\n            k_prev_fp8, v_prev_fp8 = self.quant_method.dequantize_prev_kv(\n                k_fp4[kv_indices],\n                k_scales[kv_indices],\n                v_fp4[kv_indices],\n                v_scales[kv_indices],\n                global_layer_id,\n            )\n            dq_k[kv_indices] = k_prev_fp8\n            dq_v[kv_indices] = v_prev_fp8\n'''
decode_new = '''            kv_indices = req_to_token[req_idx, :seq_len]\n            if hasattr(self.quant_method, \"dequantize_prev_kv_into_workspace\"):\n                # MTP-PP-NVFP4 direct decode dequant into shared workspace\n                self.quant_method.dequantize_prev_kv_into_workspace(\n                    k_fp4,\n                    k_scales,\n                    v_fp4,\n                    v_scales,\n                    kv_indices,\n                    global_layer_id,\n                    dq_k,\n                    dq_v,\n                    dst_indices=kv_indices,\n                )\n            else:\n                k_prev_fp8, v_prev_fp8 = self.quant_method.dequantize_prev_kv(\n                    k_fp4[kv_indices],\n                    k_scales[kv_indices],\n                    v_fp4[kv_indices],\n                    v_scales[kv_indices],\n                    global_layer_id,\n                )\n                dq_k[kv_indices] = k_prev_fp8\n                dq_v[kv_indices] = v_prev_fp8\n'''

if decode_marker not in mtext:
    if decode_old not in mtext:
        raise SystemExit(
            "ERROR: NVFP4 decode workspace source shape changed; refusing blind patch"
        )
    mtext = mtext.replace(decode_old, decode_new, 1)

# Refuse to write syntactically invalid image-time source.
compile(qtext, str(QPATH), "exec")
compile(mtext, str(MPATH), "exec")
QPATH.write_text(qtext)
MPATH.write_text(mtext)

print(
    "[MTP-PP-NVFP4-DEQUANT] installed direct-to-workspace sequential K/V dequant patch"
)
