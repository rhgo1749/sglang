# WIP checkpoint: Qwen3.8 MTP CUDA2 sidecar

This branch is an intermediate checkpoint for an experimental SGLang architecture that moves the Qwen3.8 MTP/draft path to a third GPU while keeping the target model on TP2.

## Intended topology

- Target: RTX 5070 Ti + RTX 5070 Ti, TP2
- Draft/MTP sidecar: RTX 5060 Ti, TP1
- Target hidden states are transferred to the sidecar; sidecar proposals are returned to the target for verification.
- MTP must remain enabled.

## Verified so far

- CUDA logical mapping with `--gpus '"device=0,2,1"'` gives CUDA0/1 = RTX 5070 Ti and CUDA2 = RTX 5060 Ti.
- A standalone TP1 MTP worker can be constructed on CUDA2.
- Sidecar model load succeeds.
- Independent CUDA2 request/token/KV/Mamba pools succeed.
- CUDA2 sidecar attention backend initializes successfully (`HybridLinearAttnBackend`).
- Sidecar eager runner initializes successfully.
- Qwen3.8 MTP body runs far enough to enter real attention computation on CUDA2.
- A process-wide RoPE cache caused the sidecar to inherit `cos_sin_cache` from target CUDA0. This was fixed experimentally by deep-copying the RoPE module for the sidecar and moving only the private copy to CUDA2.
- After RoPE isolation, `q_norm`, `k_norm`, `positions`, hidden states, request indices, output cache locations and RoPE cache were all confirmed on CUDA2.
- The sidecar forward then advanced past attention into the layer communicator.

## Current blocker

The current failure occurs in the layer communicator during:

`attention_tensor_model_parallel_all_reduce()` -> `get_attn_tp_group().all_reduce()`

The sidecar is TP1, but `draft_tp_context()` only patches the global tensor-parallel group (`_TP`). The separate attention tensor-parallel global (`_ATTN_TP`) remains pointed at the target TP2 NCCL group. As a result, a CUDA2 sidecar tensor is submitted to the target TP2 communicator and NCCL fails with `ncclUnhandledCudaError`.

Exact-image inspection confirmed:

- `draft_tp_context(tp_group)` only wraps `patch_tensor_parallel_group(tp_group)`.
- `get_attn_tp_group()` returns the separate `_ATTN_TP` global.
- Normal initialization may alias `_ATTN_TP = _TP`, but the draft context does not patch `_ATTN_TP` when `_TP` is temporarily replaced.

## Next experiment

Temporarily patch `_ATTN_TP` to the same singleton coordinator used by the TP1 sidecar only for the sidecar forward, then restore the target `_ATTN_TP` immediately afterward. The goal is to complete the first real CUDA2 MTP eager prefill before moving on to persistent request mirroring, decode, graph capture, rank relay, or removing the colocated target-side draft allocation.

## Important implementation notes

- Do not pass target `Req` objects directly to sidecar request pools; pool allocation mutates request and Mamba ownership fields. Use isolated/persistent sidecar request clones.
- Do not mutate the target `ScheduleBatch`; sidecar batches must be shallow-copied and CUDA tensors explicitly remapped to CUDA2.
- The sidecar needs independent request/token/KV/Mamba pools.
- The sidecar TP1 model must keep its own full embedding and lm_head. A local experimental patch loads base embedding/lm_head weights into the TP1 MTP model and keeps the lm_head on the checkpoint's native ModelOpt quantization path while leaving the MTP body BF16.
- Do not move the cached RoPE module in place because it is shared process-wide; use a private sidecar copy.
- Keep `--mm-feature-transport cpu`; CUDA IPC across these GPUs has already been problematic.

This document intentionally records a WIP state rather than a production-ready implementation.