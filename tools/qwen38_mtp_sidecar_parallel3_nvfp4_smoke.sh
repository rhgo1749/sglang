#!/usr/bin/env bash
set -euo pipefail

REPO="${HOME}/projects/sglang-fork"
PATCH_DIR="${HOME}/projects/sglang-patches"
CONTAINER="sglang-qwen38-test"
IMAGE="lmsysorg/sglang:qwen38-27b"
MODEL="RadixArk/Qwen3.8-27B-NVFP4"
MEM_FRACTION_STATIC="${MTP_P3_MEM_FRACTION_STATIC:-0.80}"
MAX_RUNNING="${MTP_P3_MAX_RUNNING:-3}"
MAX_MAMBA="${MTP_P3_MAX_MAMBA:-10}"
OUTPUT_TOKENS="${MTP_P3_OUTPUT_TOKENS:-64}"
REQUEST_TIMEOUT="${MTP_P3_REQUEST_TIMEOUT:-180}"

cd "$REPO"

echo '=== PREFLIGHT EXACT IMAGE NVFP4 TARGET ABI ==='
HELP_OUT="$(mktemp)"
if ! docker run --rm "$IMAGE" python3 -m sglang.launch_server --help >"$HELP_OUT" 2>&1; then
  cat "$HELP_OUT"
  exit 1
fi
for needle in '--kv-cache-dtype' '--prefill-attention-backend' '--decode-attention-backend' '--speculative-draft-attention-backend'; do
  if ! grep -q -- "$needle" "$HELP_OUT"; then
    echo "ERROR: exact image does not expose $needle"
    grep -E -- 'kv-cache-dtype|attention-backend|speculative-draft' "$HELP_OUT" | head -80 || true
    exit 1
  fi
done
if ! grep -q 'nvfp4' "$HELP_OUT" || ! grep -q 'trtllm_mha' "$HELP_OUT"; then
  echo 'ERROR: exact image does not advertise nvfp4 + trtllm_mha'
  grep -E -- 'kv-cache-dtype|attention-backend|trtllm_mha|nvfp4' "$HELP_OUT" | head -100 || true
  exit 1
fi
echo 'target: NVFP4 KV (FlashInfer prefill + TRTLLM-MHA decode, page_size=64)'
echo 'CUDA2 draft: FP8 E4M3 KV (FlashInfer multi-step, private page_size=1)'

echo '=== APPLY AUTHORITATIVE CUTOVER ==='
python3 tools/qwen38_mtp_sidecar_cutover.py --commit
python3 tools/qwen38_mtp_cutover_pp_hotfix.py
python3 tools/qwen38_mtp_cutover_mamba_tracking_hotfix.py

echo '=== HOTFIX BATCH-AWARE CUDA2 STATE ==='
python3 tools/qwen38_mtp_cutover_parallel3_hotfix.py --commit

echo '=== HOTFIX PARALLEL POOL GATE ==='
python3 tools/qwen38_mtp_cutover_pool_gate_hotfix.py --commit

echo '=== HOTFIX NVFP4 TARGET / FP8 CUDA2 DRAFT ==='
python3 tools/qwen38_mtp_cutover_nvfp4_target_hotfix.py --commit

echo '=== HOTFIX CUDA2 SIDECAR PAGE_SIZE=1 ==='
python3 tools/qwen38_mtp_cutover_sidecar_page1_hotfix.py --commit

echo '=== RECREATE SERVER: NVFP4 TARGET KV, FP8 DRAFT KV, LOGICAL 256K x 3 ==='
docker rm -f "$CONTAINER" >/dev/null 2>&1 || true

docker run -d \
  --name "$CONTAINER" \
  --gpus '"device=0,2,1"' \
  --ipc=host \
  --shm-size 32g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
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
    --kv-cache-dtype nvfp4 \
    --attention-backend flashinfer \
    --prefill-attention-backend flashinfer \
    --decode-attention-backend trtllm_mha \
    --speculative-draft-attention-backend flashinfer \
    --mem-fraction-static "$MEM_FRACTION_STATIC" \
    --max-running-requests "$MAX_RUNNING" \
    --max-mamba-cache-size "$MAX_MAMBA" \
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

STARTED="$(docker inspect -f '{{.State.StartedAt}}' "$CONTAINER")"
echo "StartedAt: $STARTED"

dump_failure() {
  echo '=== FAILURE SUMMARY ==='
  docker logs --since "$STARTED" "$CONTAINER" 2>&1 | \
    grep -E 'MTP-CUTOVER|NVFP4|FP4|FP8|KV Cache|kv.cache|kv_cache|Mamba Cache|Memory pool end|max_total_num_tokens|available_gpu_mem|Scheduler hit an exception|Traceback|Exception|RuntimeError|ValueError|AssertionError|AttributeError|CUDA error|illegal memory|out of memory|NCCL|Connection closed' | \
    tail -620 || true
  echo '=== FAILURE TAIL ==='
  docker logs --since "$STARTED" "$CONTAINER" 2>&1 | tail -420 || true
  echo '=== CONTAINER STATUS ==='
  docker inspect -f 'running={{.State.Running}} status={{.State.Status}} exit={{.State.ExitCode}}' "$CONTAINER" || true
}

if ! timeout 300 bash -c '
while true; do
  if curl -fsS http://127.0.0.1:30000/model_info >/dev/null 2>&1; then exit 0; fi
  if [ "$(docker inspect -f "{{.State.Running}}" sglang-qwen38-test 2>/dev/null)" != "true" ]; then exit 2; fi
  sleep 1
done
'; then
  echo 'SERVER DID NOT BECOME READY'
  dump_failure
  exit 1
fi

echo '=== STARTUP GATES ==='
START_LOG="$(mktemp)"
docker logs --since "$STARTED" "$CONTAINER" >"$START_LOG" 2>&1 || true
grep -E 'MTP-CUTOVER|NVFP4|FP4|FP8|KV Cache|Mamba Cache|Memory pool end|max_total_num_tokens|available_gpu_mem|Capture target verify CUDA graph|TRTLLM' "$START_LOG" | tail -420 || true

if ! grep -q '\[MTP-CUTOVER-KV\] target=nvfp4 CUDA2 draft=fp8_e4m3' "$START_LOG"; then
  echo 'ERROR: CUDA2 private FP8 draft-KV override did not activate'
  dump_failure
  exit 1
fi
if ! grep -q 'draft_kv_dtype=torch.float8_e4m3fn' "$START_LOG"; then
  echo 'ERROR: CUDA2 draft runner did not resolve FP8 E4M3 KV'
  dump_failure
  exit 1
fi
if ! grep -Eq '\[MTP-CUTOVER-PAGE\].*draft_page_size=1.*allocator=.*TokenToKVPoolAllocator' "$START_LOG"; then
  echo 'ERROR: CUDA2 draft did not isolate to page_size=1 token allocator'
  dump_failure
  exit 1
fi

TARGET_TOKENS="$(sed -nE 's/.*MTP-CUTOVER-POOL.*target_rank=0 target_tokens=([0-9]+).*/\1/p' "$START_LOG" | tail -1)"
SIDE_TOKENS="$(sed -nE 's/.*MTP-CUTOVER-POOL.*CUDA2 side_tokens=([0-9]+).*/\1/p' "$START_LOG" | tail -1)"
if [[ -z "$TARGET_TOKENS" || -z "$SIDE_TOKENS" ]]; then
  echo 'ERROR: failed to parse token pools'
  dump_failure
  exit 1
fi
printf 'target_kv=nvfp4\ntarget_page_size=64\ndraft_kv=fp8_e4m3\ndraft_page_size=1\ntarget_tokens=%s\nside_tokens=%s\nlogical_context_per_request=262144\nmax_running_requests=%s\n' \
  "$TARGET_TOKENS" "$SIDE_TOKENS" "$MAX_RUNNING"

POOL_MIN="$TARGET_TOKENS"
if (( SIDE_TOKENS < POOL_MIN )); then POOL_MIN="$SIDE_TOKENS"; fi
RESERVE=8192
PER_PROMPT=$(( (POOL_MIN - RESERVE) / MAX_RUNNING ))
if (( PER_PROMPT > 65520 )); then PER_PROMPT=65520; fi
if (( PER_PROMPT < 4096 )); then
  echo "ERROR: shared pool too small for parallel stress: $POOL_MIN"
  exit 1
fi

echo "=== PARALLEL-${MAX_RUNNING} STRESS: ${PER_PROMPT} TOKENS EACH + ${OUTPUT_TOKENS} (timeout=${REQUEST_TIMEOUT}s) ==="
for i in $(seq 0 $((MAX_RUNNING - 1))); do
  python3 - "$i" "$PER_PROMPT" "$OUTPUT_TOKENS" <<'PY'
import json, sys
idx, n, out = map(int, sys.argv[1:])
with open(f'/tmp/qwen38-nvfp4-p3-{idx}.json', 'w') as f:
    json.dump({
        'input_ids': [1000 + idx] * n,
        'sampling_params': {'temperature': 0, 'max_new_tokens': out, 'ignore_eos': True},
    }, f, separators=(',', ':'))
PY
done

REQ_SINCE="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
PIDS=()
START_NS="$(date +%s%N)"
for i in $(seq 0 $((MAX_RUNNING - 1))); do
  (
    curl --max-time "$REQUEST_TIMEOUT" -sS \
      -o "/tmp/qwen38-nvfp4-p3-${i}-response.json" \
      -w '%{http_code}' \
      http://127.0.0.1:30000/generate \
      -H 'Content-Type: application/json' \
      --data-binary "@/tmp/qwen38-nvfp4-p3-${i}.json" \
      > "/tmp/qwen38-nvfp4-p3-${i}-http.txt"
  ) &
  PIDS+=("$!")
done

RC=0
for pid in "${PIDS[@]}"; do
  wait "$pid" || RC=1
done
END_NS="$(date +%s%N)"

sleep 1
REQ_LOG=/tmp/qwen38-nvfp4-p3-runtime.log
docker logs --since "$REQ_SINCE" "$CONTAINER" >"$REQ_LOG" 2>&1 || true

set +e
python3 - "$MAX_RUNNING" "$START_NS" "$END_NS" "$PER_PROMPT" "$OUTPUT_TOKENS" <<'PY'
import json, pathlib, sys
nr, start_ns, end_ns, expected_prompt, expected_output = map(int, sys.argv[1:])
wall = (end_ns - start_ns) / 1e9
print(f'parallel_wall_sec={wall:.3f}')
all_ok = True
sum_prompt = sum_completion = 0
for i in range(nr):
    http_path = pathlib.Path(f'/tmp/qwen38-nvfp4-p3-{i}-http.txt')
    http = http_path.read_text().strip() if http_path.exists() else ''
    print(f'req{i}_http={http}')
    try:
        d = json.load(open(f'/tmp/qwen38-nvfp4-p3-{i}-response.json'))
        m = d.get('meta_info', {})
        p = int(m.get('prompt_tokens') or 0)
        c = int(m.get('completion_tokens') or 0)
        print(f'req{i}_prompt_tokens={p}')
        print(f'req{i}_completion_tokens={c}')
        print(f'req{i}_finish_reason={m.get("finish_reason")}')
        sum_prompt += p
        sum_completion += c
        all_ok &= http == '200' and p == expected_prompt and c == expected_output
    except Exception as e:
        print(f'req{i}_parse_error={e!r}')
        all_ok = False
print(f'aggregate_prompt_tokens={sum_prompt}')
print(f'aggregate_completion_tokens={sum_completion}')
print(f'aggregate_output_tps={sum_completion / wall:.2f}')
print(f'parallel3_pass={all_ok}')
if not all_ok:
    raise SystemExit(1)
PY
PY_RC=$?
set -e

if [[ $RC -ne 0 || $PY_RC -ne 0 ]]; then
  echo 'PARALLEL REQUEST FAILURE'
  dump_failure
  exit 1
fi

echo '=== PARALLEL CUTOVER LOG ==='
grep -E 'MTP-CUTOVER-(REQ|PREFILL|DRAFT|EXTEND|KV|PAGE)|Prefill batch|Decode batch|Scheduler hit an exception|Traceback|CUDA error|illegal memory|out of memory' "$REQ_LOG" | tail -520 || true

echo '=== CONTAINER STATUS ==='
docker inspect -f 'running={{.State.Running}} status={{.State.Status}} exit={{.State.ExitCode}}' "$CONTAINER"

echo 'MTP NVFP4-TARGET / FP8-DRAFT PARALLEL-3 SMOKE COMPLETE'
