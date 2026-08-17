#!/usr/bin/env python3
from pathlib import Path

PATH = Path("/sgl-workspace/sglang/python/sglang/srt/layers/quantization/fp4_kv_cache_quant_method.py")
text = PATH.read_text()

old = '''        k_bf16 = NVFP4KVQuantizeUtil.dequantize(\n            k_fp4.view(torch.uint8), k_scales, cur_k_scale\n        )\n        v_bf16 = NVFP4KVQuantizeUtil.dequantize(\n            v_fp4.view(torch.uint8), v_scales, cur_v_scale\n        )\n        return k_bf16.to(torch.float8_e4m3fn), v_bf16.to(torch.float8_e4m3fn)\n'''
new = '''        # Keep peak temporary memory bounded during very long prefix dequant.\n        # The old order materialized both BF16 K and BF16 V before converting\n        # either to FP8, so the second FP8 cast could see K_BF16 + V_BF16 +\n        # K_FP8 live at once.  Convert K immediately and release its BF16\n        # temporary before materializing V.  Returned tensors and numerical\n        # formats are unchanged; only temporary lifetimes are shortened.\n        k_bf16 = NVFP4KVQuantizeUtil.dequantize(\n            k_fp4.view(torch.uint8), k_scales, cur_k_scale\n        )\n        k_fp8 = k_bf16.to(torch.float8_e4m3fn)\n        del k_bf16\n\n        v_bf16 = NVFP4KVQuantizeUtil.dequantize(\n            v_fp4.view(torch.uint8), v_scales, cur_v_scale\n        )\n        v_fp8 = v_bf16.to(torch.float8_e4m3fn)\n        del v_bf16\n\n        return k_fp8, v_fp8\n'''

marker = "# Keep peak temporary memory bounded during very long prefix dequant."
if marker in text:
    print("[MTP-PP-NVFP4-DEQUANT] sequential dequant already installed")
elif old not in text:
    raise SystemExit("ERROR: NVFP4 dequant source shape changed; refusing blind patch")
else:
    text = text.replace(old, new, 1)
    PATH.write_text(text)
    print("[MTP-PP-NVFP4-DEQUANT] installed sequential K/V dequant temporary lifetime patch")
