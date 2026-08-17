#!/usr/bin/env bash
set -euo pipefail

REPO="${HOME}/projects/sglang-fork"
PATCH_DIR="${HOME}/projects/sglang-patches"
CONTAINER="${MTP_AB_CONTAINER:-sglang-qwen38-test}"
IMAGE="lmsysorg/sglang:qwen38-27b"
MODEL="RadixArk/Qwen3.8-27B-NVFP4"
PROMPT_TOKENS="${MTP_AB_PROMPT_TOKENS:-27000}"
OUTPUT_TOKENS="${MTP_AB_OUTPUT_TOKENS:-512}"
MIN_ACCEPT="${MTP_AB_MIN_ACCEPT:-2.0}"
MEM_FRACTION_STATIC="${MTP_AB_MEM_FRACTION_STATIC:-0.80}"
RESULTS="$(mktemp)"
trap 'rm -f "$RESULTS"' EXIT

cd "$REPO"

echo '=== TARGET VERIFY A/B PREFLIGHT ==='
python3 -m py_compile python/sglang/srt/speculative/eagle_worker_v2.py
python3 -m py_compile "$PATCH_DIR/eagle_worker_v2.sidecar-pool-probe.py"
python3 -m py_compile "$PATCH_DIR/qwen3_5_mtp.sidecar.py"
bash -n tools/qwen38_mtp_sidecar_benchmark.sh
echo 'target verify A/B syntax: OK'

wait_ready() {
  local started="$1"
  if ! timeout 300 bash -c '
    while true; do
      if curl -fsS http://127.0.0.1:30000/model_info >/dev/null 2>&1; then exit 0; fi
      if [ "$(docker inspect -f "{{.State.Running}}" sglang-qwen38-test 2>/dev/null)" != "true" ]; then exit 2; fi
      sleep 1
    done
  '; then
    echo 'SERVER DID NOT BECOME READY'
    docker logs --since "$started" "$CONTAINER" 2>&1 | tail -320 || true
    return 1
  fi
}

run_case() {
  local name="$1"
  local kv_dtype="$2"
  local decode_backend="$3"
  local page_size="$4"
  local started startup bench target_tokens side_tokens accept tps

  echo
  echo "=== CASE ${name}: target_kv=${kv_dtype} target_decode=${decode_backend} target_page=${page_size} max_running=1 ==="
  docker rm -f "$CONTAINER" >/dev/null 2>&1 || true

  docker run -d \
    --name "$CONTAINER" \
    --gpus '"device=0,2,1"' \
    --ipc=host \
    --shm-size 32g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -e SGLANG_MTP_CUTOVER_MIN_POOL_TOKENS=65536 \
    -p 30000:30000 \
    -v sglang-hf-cache:/root/.cache/huggingface \
    -v "$PATCH_DIR/eagle_worker_v2.sidecar-pool-probe.py:/sgl-workspace/sglang/python/sglang/srt/speculative/eagle_worker_v2.py:ro" \
    -v "$PATCH_DIR/qwen3_5_mtp.sidecar.py:/sgl-workspace/sglang/python/sglang/srt/models/qwen3_5_mtp.py:ro" \
    "$IMAGE" \
    python3 -m sglang.launch_server \
      --model-path "$MODEL" \
      --tp 2 \
      --trust-remote-code \
      --context-length 262144 \
      --kv-cache-dtype "$kv_dtype" \
      --page-size "$page_size" \
      --attention-backend flashinfer \
      --prefill-attention-backend flashinfer \
      --decode-attention-backend "$decode_backend" \
      --speculative-draft-attention-backend flashinfer \
      --mem-fraction-static "$MEM_FRACTION_STATIC" \
      --max-running-requests 1 \
      --max-mamba-cache-size 8 \
      --mamba-ssm-dtype bfloat16 \
      --mamba-radix-cache-strategy extra_buffer_lazy \
      --disable-radix-cache \
      --disable-custom-all-reduce \
      --mm-feature-transport cpu \
      --chunked-prefill-size 2048 \
      --speculative-algorithm EAGLE \
      --speculative-num-steps 3 \
      --speculative-eagle-topk 1 \
      --speculative-num-draft-tokens 4 \
      --reasoning-parser qwen3 \
      --tool-call-parser qwen3_coder \
      --host 0.0.0.0 \
      --port 30000 >/dev/null

  started="$(docker inspect -f '{{.State.StartedAt}}' "$CONTAINER")"
  wait_ready "$started"

  startup="$(mktemp)"
  bench="$(mktemp)"
  docker logs --since "$started" "$CONTAINER" >"$startup" 2>&1 || true

  if ! grep -Eq '\[MTP-CUTOVER-PAGE\].*draft_page_size=1.*allocator=TokenToKVPoolAllocator' "$startup"; then
    echo "ERROR: ${name} CUDA2 draft did not stay on page1 TokenToKVPoolAllocator"
    grep -E 'MTP-CUTOVER-(PAGE|KV|ATTN|POOL)|KV Cache|Memory pool end|Traceback|RuntimeError' "$startup" | tail -220 || true
    return 1
  fi
  if ! grep -Eq '\[MTP-CUTOVER-ATTN\].*decode=FlashInferMultiStepDraftBackend.*extend=FlashInferAttnBackend' "$startup"; then
    echo "ERROR: ${name} CUDA2 draft did not resolve FlashInfer decode+extend"
    grep -E 'MTP-CUTOVER-(PAGE|KV|ATTN|POOL)|Traceback|RuntimeError' "$startup" | tail -220 || true
    return 1
  fi

  target_tokens="$(sed -nE 's/.*MTP-CUTOVER-POOL.*target_rank=0 target_tokens=([0-9]+).*/\1/p' "$startup" | tail -1)"
  side_tokens="$(sed -nE 's/.*MTP-CUTOVER-POOL.*CUDA2 side_tokens=([0-9]+).*/\1/p' "$startup" | tail -1)"
  echo "${name}_target_tokens=${target_tokens:-unknown}"
  echo "${name}_side_tokens=${side_tokens:-unknown}"
  grep -E 'server_args=ServerArgs|KV Cache is allocated|MTP-CUTOVER-(KV|PAGE|ATTN|POOL)' "$startup" | tail -24 || true

  if ! MTP_BENCH_PROMPT_TOKENS="$PROMPT_TOKENS" \
       MTP_BENCH_OUTPUT_TOKENS="$OUTPUT_TOKENS" \
       bash tools/qwen38_mtp_sidecar_benchmark.sh >"$bench" 2>&1; then
    echo "ERROR: benchmark failed for ${name}"
    cat "$bench"
    return 1
  fi

  grep -E '^(HTTP|prompt_tokens|completion_tokens|wall_sec|end_to_end_output_tps|verify_iterations|accept_mean|accept_median|accept_min|accept_max|accepted_tokens_total|effective_tokens_per_verify)=' "$bench" || true
  accept="$(sed -nE 's/^accept_mean=([0-9.]+)$/\1/p' "$bench" | tail -1)"
  tps="$(sed -nE 's/^end_to_end_output_tps=([0-9.]+)$/\1/p' "$bench" | tail -1)"
  if [[ -z "$accept" || -z "$tps" ]]; then
    echo "ERROR: failed to parse benchmark result for ${name}"
    tail -180 "$bench"
    return 1
  fi

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$name" "$kv_dtype" "$decode_backend" "$page_size" "$accept" "$tps" "${target_tokens:-0}" >>"$RESULTS"
  rm -f "$startup" "$bench"
}

# N1: same target format/backend as the failing production candidate, but restore
# historical single-request scheduler/Mamba sizing. If this alone recovers,
# max_running=3 state is the culprit rather than NVFP4.
run_case N1 nvfp4 trtllm_mha 64

# T1: keep TRTLLM-MHA/page64 but change only target KV storage to FP8.
# Draft remains FP8 + FlashInfer + page1 via the dedicated draft backend flag.
run_case T1 fp8_e4m3 trtllm_mha 64

# F1: keep target FP8 but restore the historical FlashInfer/page1 target path.
run_case F1 fp8_e4m3 flashinfer 1

# H1: exact historical-style control: automatic target KV dtype + FlashInfer/page1.
run_case H1 auto flashinfer 1

echo
echo '=== TARGET VERIFY A/B RESULTS ==='
printf 'case\ttarget_kv\ttarget_decode\tpage\taccept_mean\toutput_tps\ttarget_tokens\n'
cat "$RESULTS"

get_accept() {
  awk -F '\t' -v k="$1" '$1==k {print $5}' "$RESULTS"
}
pass_accept() {
  awk -v a="$1" -v m="$MIN_ACCEPT" 'BEGIN { exit !(a+0 >= m+0) }'
}

N1="$(get_accept N1)"
T1="$(get_accept T1)"
F1="$(get_accept F1)"
H1="$(get_accept H1)"

echo "acceptance_gate=${MIN_ACCEPT}"
if pass_accept "$N1"; then
  echo 'A/B VERDICT: NVFP4 itself can accept at max_running=1; investigate max_running=3 / multi-request sidecar state next.'
elif pass_accept "$T1"; then
  echo 'A/B VERDICT: target NVFP4 KV is the acceptance breaker; FP8 target works even through TRTLLM-MHA/page64.'
elif pass_accept "$F1"; then
  echo 'A/B VERDICT: TRTLLM-MHA/page64 target verify path is the acceptance breaker; FP8 + FlashInfer/page1 recovers.'
elif pass_accept "$H1"; then
  echo 'A/B VERDICT: explicit FP8 target is also harmful; historical AUTO + FlashInfer/page1 is the only recovered control.'
else
  echo 'A/B VERDICT: all current-code controls failed; regression is in cutover state/allocation logic added after the historical good baseline, not just target KV/backend.'
fi

echo 'TARGET VERIFY A/B COMPLETE'
