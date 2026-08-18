# Local consumer-GPU Qwen3.8 serving experiments (2026-08-18)

This note records the final results from testing Qwen3.8-27B native NVFP4 + native MTP on a 3-GPU consumer workstation, and comparing that setup with llama.cpp GGUF serving on the same machine.

## Hardware

- RTX 5070 Ti 16 GB x2
- RTX 5060 Ti 16 GB x1
- Consumer PCIe multi-GPU topology, no NVLink

Physical GPU indices used during the experiments:

- GPU0: RTX 5070 Ti
- GPU1: RTX 5060 Ti
- GPU2: RTX 5070 Ti

## SGLang model and validated PP3 baseline

Native model:

- `gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090`

Final correctness-validated SGLang production baseline during the experiment:

- PP = 3
- GPU order = physical `0,2,1`
- PP0 = RTX 5070 Ti
- PP1 = RTX 5070 Ti
- PP2 = RTX 5060 Ti + native MTP worker
- partition = `23,28,13`
- context = 262144
- max total tokens = 524288
- max running requests = 2
- max Mamba cache size = 2
- mem fraction static = 0.84
- NVFP4 KV cache
- FlashInfer attention/prefill
- TRTLLM MHA decode
- page size 64
- native MTP / NEXTN: 3 steps, top-k 1, 4 draft tokens
- CUDA Graph completely disabled
- radix cache disabled

This configuration passed the 2x256K capacity gate and semantic boot control.

## Native MTP benefit

A/B timing with the validated PP3 topology:

- MTP ON: 98.922 s
- MTP OFF: 109.844 s
- measured MTP-on improvement: about 11.04%

Conclusion: native MTP was beneficial and should not be disabled for this configuration.

## CUDA Graph investigation

All relevant CUDA Graph paths were investigated and closed for production use:

- target verify CUDA Graph: correctness unsafe
- prefill BCG: correctness unsafe
- EAGLE draft-decode CUDA Graph: post-long correctness unsafe
- EAGLE draft-extend CUDA Graph: correctness stable, but about 5.8% slower than eager

Final decision: CUDA Graph OFF for the correctness baseline.

## MTP placement experiments

The PP bridge used by this fork colocates the native MTP draft worker on PP-last. Therefore changing the PP-last physical GPU also changes where the last target shard, sampling/acceptance, and MTP work run.

Measured placement cases:

- Current topology, MTP on PP-last RTX 5060 Ti, partition `23,28,13`: 98.922 s
- MTP OFF: 109.844 s
- PP-last moved to an RTX 5070 Ti while the 5060 Ti became PP1: 107.629 s

The last case was not a pure draft-only relocation; it also changed the target-stage placement, so it did not demonstrate that a 5070-hosted MTP worker was intrinsically slower.

A pure PP3 MTP sidecar relocation would require substantial integration work. The existing authoritative sidecar prototype in `eagle_worker_v2.py` was designed around a different PP=1/TP2 topology and was not directly usable for the current PP3 setup.

## PP3 partition capacity findings

The 2x256K requirement is 524288 total tokens.

Important measured boundaries:

### PP0 / RTX 5070 Ti

- 23 target layers: 524288 tokens possible
- 24 target layers: 470912 tokens
- 26 target layers: 429376 tokens

Therefore PP0 could not be increased beyond 23 layers while preserving the fixed 2x256K pool.

### PP2 + MTP

Measured examples with MTP on the last stage:

- 11 target layers: 524288 possible
- 13 target layers: 524288 possible
- 16 target layers: 524288 possible in a separate placement test
- 19 target layers: 438336
- 20 target layers: 408768
- 21 target layers: 332800

Examples that failed:

- `23,22,19` -> 438336
- `23,21,20` -> 408768
- `23,20,21` -> 332800
- `24,29,11` -> 470912 because PP0 became the limiting stage

These measurements show why `23,28,13` is a strong practical point under the fixed 2x256K requirement: PP0 cannot take another layer, while the MTP-last stage must remain relatively light.

The partition sweep was closed after this point; no further one-layer microtuning was considered worthwhile.

## Dual-5070 SGLang PP2 / single-request experiment

A final attempt was made to remove the 5060 Ti and run:

- RTX 5070 Ti x2
- PP = 2
- max running requests = 1
- context / pool = 262144
- native MTP on PP-last
- initial partition = `31,33`

The server failed during MTP draft-worker initialization before the actual 262K KV capacity became the problem.

Relevant result:

- PP0 with 31 target layers profiled 262144 local tokens successfully
- PP1 target shard loaded with about 9.89 GB used
- during MTP initialization, creation of the draft `lm_head` required another ~2.37 GiB
- PP1 then hit CUDA OOM

This demonstrated a structural memory problem for the dual-16GB PP2 + colocated native-MTP layout. Further `30,34`, `29,35`, etc. partition probing was intentionally stopped rather than continuing microtuning.

## llama.cpp comparison

The comparison was operational rather than a strict engine-only apples-to-apples model comparison, because the llama.cpp runs used GGUF artifacts while SGLang used the native checkpoint.

### llama.cpp dual-5070 / parallel=1

Configuration used for the key baseline:

- RTX 5070 Ti x2 only
- target and MTP both on the two 5070 Ti GPUs
- tensor split `1,1`
- parallel = 1
- context = 262144
- Q8_0 KV and draft KV
- MTP max draft tokens = 4

Measured `4096 prompt + 1024 completion`, two requests executed serially:

- request 0 wall: 9.368 s (included extra router/model activation overhead)
- request 1 wall: 6.692 s
- serial-2 wall: 16.061 s
- engine prompt throughput: ~2724.7 tok/s
- engine generation throughput: ~201.4 tok/s

Measured `262000 prompt + 8 completion`, two requests serially:

- request 0 wall: 200.136 s
- request 1 wall: 202.278 s
- serial-2 wall: 402.423 s
- engine prompt throughput: ~1311.6 tok/s
- engine generation throughput: ~61.0 tok/s

Both exact-token checks passed.

### llama.cpp 3-GPU / parallel=3

The main router configuration used:

- CUDA0 = RTX 5070 Ti
- CUDA1 = RTX 5070 Ti
- CUDA2 = RTX 5060 Ti
- target devices = CUDA0,CUDA1,CUDA2
- target tensor split = `1,1,0.3`
- MTP draft device = CUDA2
- parallel = 3
- context = 262144 total router context

Two concurrent `4096 + 1024` requests:

- wall = 12.887 s
- effective completion throughput = 158.914 tok/s
- per-request generation rates were about 113 to 130 tok/s

Three concurrent `4096 + 1024` requests:

- wall = 14.923 s
- effective completion throughput = 205.856 tok/s
- per-request generation rates: 109.277, 118.229, and 95.962 tok/s
- all exact-token checks passed

Thus 3-way concurrency increased total completion throughput by about 29.5% versus the measured 2-way concurrent case, while only increasing wall time by about 15.8%.

A two-request `262000 + 8` long-context test in the parallel=3 router failed with HTTP 500 `Context size has been exceeded.` after ~225 s. That time is not a successful performance result and must not be treated as a 1.787x long-context speedup. The router's 262144 total context is shared across its parallel slots, so full 262K requests are incompatible with that parallel=3 configuration.

## Final operational conclusion

For this consumer-GPU workstation:

### llama.cpp parallel=1 / dual RTX 5070 Ti

Best fit for:

- single-request latency
- full 256K context
- long-context agent work
- freeing the RTX 5060 Ti for other workloads

### llama.cpp parallel=3 / 5070 + 5070 + 5060

Best fit for:

- several short or medium-context agents at once
- higher aggregate throughput
- three simultaneous workers

The measured 3-way `4096+1024` workload reached ~205.9 effective completion tok/s, versus ~158.9 tok/s for two concurrent requests in the same 3-GPU router.

### SGLang

The native-MTP PP3 setup was made correct and functional for 2x256K serving, but on this particular set of 16 GB consumer GPUs it required substantial PP/MTP memory work and remained much less attractive operationally than llama.cpp for the intended local-agent workloads.

The final decision after these experiments was to stop further SGLang tuning on this workstation and keep the fork as an experiment/reference record.

## Closed investigations

The following areas were intentionally considered closed at the end of the experiment:

- CUDA Graph tuning for native MTP
- generic Mamba clear/lifecycle experiments
- MTP-on-5070 PP3 layer-by-layer partition sweeps
- PP3 broad partition sweep
- dual-5070 PP2/native-MTP partition microtuning
- further SGLang production tuning for this local consumer-GPU host
