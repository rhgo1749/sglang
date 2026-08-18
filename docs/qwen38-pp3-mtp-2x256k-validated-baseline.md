# Qwen3.8 PP3 + native MTP + NVFP4: validated 2x256K baseline

Validated on 2026-08-18 with `gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090` using the local three-GPU PP layout.

## Frozen structural settings

- PP: 3
- TP: 1
- layer partition: `23,28,13`
- parallel requests: 2
- context length: 262144
- max total tokens: 524288
- max running requests: 2
- max Mamba cache size: 2
- static memory fraction: 0.84
- KV cache: NVFP4
- page size: 64
- MTP: NEXTN, steps=3, topk=1, draft tokens=4
- prefill backend: FlashInfer
- decode backend: TRTLLM MHA
- radix cache: disabled
- FlashInfer autotune: disabled

Do not change the partition, pool, static fraction, or Mamba/request slot count as a performance tweak. They are part of the validated memory/correctness geometry.

## Validated performance settings

- chunked prefill: 1024
- decode CUDA Graph: `full`, BS `[1,2]`
- prefill CUDA Graph: disabled
- `CUDA_LAUNCH_BLOCKING=0`

The original conservative baseline used chunked prefill 512 and CUDA Graph disabled. Keep the long gate available as the regression fallback.

## Validation results

### Full-context two-request stress

Both requests completed successfully with:

- prompt: 262000 tokens each
- completion: 8 tokens each
- HTTP 200 for both
- capacity: 524288 / 524288
- long stress: PASS

At chunked prefill 1024:

- CUDA Graph OFF: 326.991 s
- decode CUDA Graph ON: 325.681 s

This workload is prefill dominated, so decode graph changes total time only slightly.

### Decode-heavy A/B

Two parallel requests, each with 4096 prompt tokens and 1024 forced completion tokens:

- CUDA Graph OFF: 97.187 s, effective completion throughput 21.073 tok/s
- decode CUDA Graph ON: 75.884 s, effective completion throughput 26.989 tok/s
- reported speedup: 28.07%

The decode graph is therefore promoted to the serving preset.

### GPU headroom with decode CUDA Graph ON

Observed peak used/free memory:

- host GPU0 / PP0: 14442 MiB used, 1861 MiB free
- host GPU1 / PP2 (RTX 5060 Ti): 15364 MiB used, 947 MiB free
- host GPU2 / PP1: 14242 MiB used, 2061 MiB free

The PP2/5060Ti remains the tightest stage; avoid consuming this remaining headroom without a dedicated validation run.

## Important local patches in the image

- PP-aware native MTP speculative bridge
- PP-last MTP ownership and endpoint sharing
- PP-local memory capacity profiling
- FP8 row-wise MTP embedding storage
- NVFP4 previous-prefix dequant streamed directly into the persistent workspace in bounded chunks
- TARGET_VERIFY NVFP4 metadata repair
- first-verify bridge only on the final prefill chunk
- CUDA-Graph-safe MTP input-id diagnostic guard

## Production entry point

Use:

```bash
IMAGE=sglang:qwen38-27b-pp-mtp-share \
PORT=30001 \
bash tools/qwen38_gittensor_pp3_nvfp4_256k_x2_serve.sh
```

The serve preset intentionally keeps multimodal transport enabled, but native MTP PP3 multimodal correctness is not yet declared validated. Treat text serving as the current production baseline and run a dedicated multimodal gate before relying on vision inputs.
