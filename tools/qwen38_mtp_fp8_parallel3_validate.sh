#!/usr/bin/env bash
set -euo pipefail

REPO="${HOME}/projects/sglang-fork"
PATCH_DIR="${HOME}/projects/sglang-patches"
CONTAINER="${MTP_FP8_CONTAINER:-sglang-qwen38-test}"
IMAGE="${MTP_FP8_IMAGE:-lmsysorg/sglang:qwen38-27b}"
MODEL="${MTP_FP8_MODEL:-RadixArk/Qwen3.8-27B-NVFP4}"
MEM_FRACTION_STATIC="${MTP_FP8_MEM_FRACTION_STATIC:-0.80}"
MAX_RUNNING="${MTP_FP8_MAX_RUNNING:-3}"
MAX_MAMBA="${MTP_FP8_MAX_MAMBA:-8}"
P3_PROMPT="${MTP_FP8_P3_PROMPT:-25194}"
P3_OUTPUT="${MTP_FP8_P3_OUTPUT:-64}"
REQUEST_TIMEOUT="${MTP_FP8_REQUEST_TIMEOUT:-180}"
MIN_POOL="${MTP_FP8_MIN_POOL:-65536}"
MIN_ACCEPT="${MTP_FP8_MIN_ACCEPT:-3.0}"

cd "$REPO"

echo '=== FP8 PARALLEL-3 PREFLIGHT: NO SERVER TOUCH ==='
bash -n "$0"
for py in \
  tools/qwen38_mtp_sidecar_cutover.py \
  tools/qwen38_mtp_cutover_pp_hotfix.py \
  tools/qwen38_mtp_cutover_mamba_tracking_hotfix.py \
  tools/qwen38_mtp_cutover_parallel3_hotfix.py \
  tools/qwen38_mtp_cutover_pool_gate_hotfix.py \
  tools/qwen38_mtp_cutover_hybrid_fp8_draft_hotfix.py; do
  python3 -m py_compile "$py"
done
python3 -m py_compile "$PATCH_DIR/qwen3_5_mtp.sidecar.py"
bash -n tools/qwen38_mtp_sidecar_benchmark.sh
echo 'fp8 parallel3 preflight syntax: OK'

echo '=== APPLY AUTHORITATIVE SIDECAR PATCH CHAIN ==='
python3 tools/qwen38_mtp_sidecar_cutover.py --commit
python3 tools/qwen38_mtp_cutover_pp_hotfix.py
python3 tools/qwen38_mtp_cutover_mamba_tracking_hotfix.py
python3 tools/qwen38_mtp_cutover_parallel3_hotfix.py --commit
python3 tools/qwen38_mtp_cutover_pool_gate_hotfix.py --commit
python3 tools/qwen38_mtp_cutover_hybrid_fp8_draft_hotfix.py --commit

echo "=== RECREATE SERVER: FP8 TARGET + FP8 CUDA2, LOGICAL 256K x ${MAX_RUNNING}, MAMBA=${MAX_MAMBA} ==="
docker rm -f "$CONTAINER" >/dev/null 2>&1 || true

docker run -d \
  --name "$CONTAINER" \
  --gpus '"device=0,2,1"' \
  --ipc=host \
  --shm-size 32g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -e SGLANG_MTP_CUTOVER_MIN_POOL_TOKENS="$MIN_POOL" \
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
    --kv-cache-dtype fp8_e4m3 \
    --page-size 64 \
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
    --max-prefill-tokens 16384 \
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

fail_dump() {
  echo '=== FP8 VALIDATION FAILURE SUMMARY ==='
  docker logs --since "$STARTED" "$CONTAINER" 2>&1 | \
    grep -E 'MTP-CUTOVER|KV Cache|Mamba Cache|Memory pool end|max_total_num_tokens|available_gpu_mem|Capture target verify CUDA graph|Scheduler hit an exception|Traceback|Exception|RuntimeError|ValueError|AssertionError|CUDA error|illegal memory|out of memory|NCCL|Connection closed' | \
    tail -620 || true
  echo '=== FAILURE TAIL ==='
  docker logs --since "$STARTED" "$CONTAINER" 2>&1 | tail -360 || true
  echo '=== CONTAINER STATUS ==='
  docker inspect -f 'running={{.State.Running}} status={{.State.Status}} exit={{.State.ExitCode}}' "$CONTAINER" 2>/dev/null || true
}

if ! timeout 300 bash -c '
while true; do
  if curl -fsS http://127.0.0.1:30000/model_info >/dev/null 2>&1; then exit 0; fi
  if [ "$(docker inspect -f "{{.State.Running}}" sglang-qwen38-test 2>/dev/null)" != "true" ]; then exit 2; fi
  sleep 1
done
'; then
  echo 'SERVER DID NOT BECOME READY'
  fail_dump
  exit 1
fi

echo '=== STARTUP / CAPACITY GATES ==='
START_LOG="$(mktemp)"
trap 'rm -f "$START_LOG" /tmp/qwen38-fp8-p3-*.json /tmp/qwen38-fp8-p3-*-http.txt /tmp/qwen38-fp8-p3-runtime.log' EXIT
docker logs --since "$STARTED" "$CONTAINER" >"$START_LOG" 2>&1 || true
grep -E 'server_args=ServerArgs|MTP-CUTOVER|KV Cache|Mamba Cache|Memory pool end|max_total_num_tokens|available_gpu_mem|Capture target verify CUDA graph' "$START_LOG" | tail -320 || true

if ! grep -Eq 'KV Cache is allocated\. dtype: torch\.float8_e4m3fn' "$START_LOG"; then
  echo 'ERROR: target FP8 KV cache did not initialize'
  fail_dump
  exit 1
fi
if ! grep -Eq '\[MTP-CUTOVER-PAGE\].*draft_page_size=1.*allocator=TokenToKVPoolAllocator' "$START_LOG"; then
  echo 'ERROR: CUDA2 draft is not page1 TokenToKVPoolAllocator'
  fail_dump
  exit 1
fi
if ! grep -Eq '\[MTP-CUTOVER-KV\].*draft_kv_dtype=torch\.float8_e4m3fn' "$START_LOG"; then
  echo 'ERROR: CUDA2 draft is not FP8 E4M3'
  fail_dump
  exit 1
fi
if ! grep -Eq '\[MTP-CUTOVER-ATTN\].*decode=FlashInferMultiStepDraftBackend.*extend=FlashInferAttnBackend' "$START_LOG"; then
  echo 'ERROR: CUDA2 draft did not resolve FlashInfer multi-step + extend'
  fail_dump
  exit 1
fi
if ! grep -Eq '\[MTP-CUTOVER-MAMBA\].*mamba_slots=8' "$START_LOG"; then
  echo 'ERROR: CUDA2 Mamba pool did not resolve to 8 slots'
  fail_dump
  exit 1
fi

TARGET_TOKENS="$(sed -nE 's/.*MTP-CUTOVER-POOL.*target_rank=0 target_tokens=([0-9]+).*/\1/p' "$START_LOG" | tail -1)"
SIDE_TOKENS="$(sed -nE 's/.*MTP-CUTOVER-POOL.*CUDA2 side_tokens=([0-9]+).*/\1/p' "$START_LOG" | tail -1)"
if [[ -z "$TARGET_TOKENS" || -z "$SIDE_TOKENS" ]]; then
  echo 'ERROR: failed to parse target/side token pools'
  fail_dump
  exit 1
fi
POOL_MIN="$TARGET_TOKENS"
if (( SIDE_TOKENS < POOL_MIN )); then POOL_MIN="$SIDE_TOKENS"; fi

# Target uses page64. The 25,194+64 sequence occupies ceil(25,258/64)=395 pages
# = 25,280 slots per request, or 75,840 slots for three concurrent requests.
P3_REQUIRED=$(( ((P3_PROMPT + P3_OUTPUT + 63) / 64) * 64 * MAX_RUNNING ))

echo "target_tokens=${TARGET_TOKENS}"
echo "side_tokens=${SIDE_TOKENS}"
echo "shared_pool_min=${POOL_MIN}"
echo "parallel3_required_slots=${P3_REQUIRED}"
echo "logical_context_per_request=262144"
echo "max_running_requests=${MAX_RUNNING}"
echo "max_mamba_cache_size=${MAX_MAMBA}"

if (( TARGET_TOKENS < MIN_POOL || SIDE_TOKENS < MIN_POOL )); then
  echo "ERROR: pool below single-request 64K gate: target=${TARGET_TOKENS} side=${SIDE_TOKENS} min=${MIN_POOL}"
  exit 1
fi
if (( POOL_MIN < P3_REQUIRED )); then
  echo "ERROR: Mamba8 did not recover enough physical pool for ${MAX_RUNNING} x ${P3_PROMPT}+${P3_OUTPUT}: ${POOL_MIN} < ${P3_REQUIRED}"
  exit 1
fi

echo "=== PARALLEL-${MAX_RUNNING} PHYSICAL STRESS: ${P3_PROMPT} TOKENS EACH + ${P3_OUTPUT} ==="
rm -f /tmp/qwen38-fp8-p3-*.json /tmp/qwen38-fp8-p3-*-http.txt /tmp/qwen38-fp8-p3-runtime.log 2>/dev/null || true
for i in $(seq 0 $((MAX_RUNNING - 1))); do
  python3 - "$i" "$P3_PROMPT" "$P3_OUTPUT" <<'PY'
import json, sys
idx, n, out = map(int, sys.argv[1:])
with open(f'/tmp/qwen38-fp8-p3-{idx}.json', 'w') as f:
    json.dump({
        'input_ids': [1000 + idx] * n,
        'sampling_params': {
            'temperature': 0,
            'max_new_tokens': out,
            'ignore_eos': True,
        },
    }, f, separators=(',', ':'))
PY
done

REQ_SINCE="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
PIDS=()
START_NS="$(date +%s%N)"
for i in $(seq 0 $((MAX_RUNNING - 1))); do
  (
    curl --max-time "$REQUEST_TIMEOUT" -sS \
      -o "/tmp/qwen38-fp8-p3-${i}-response.json" \
      -w '%{http_code}' \
      http://127.0.0.1:30000/generate \
      -H 'Content-Type: application/json' \
      --data-binary "@/tmp/qwen38-fp8-p3-${i}.json" \
      > "/tmp/qwen38-fp8-p3-${i}-http.txt"
  ) &
  PIDS+=("$!")
done

REQ_RC=0
for pid in "${PIDS[@]}"; do
  wait "$pid" || REQ_RC=1
done
END_NS="$(date +%s%N)"
sleep 1
REQ_LOG=/tmp/qwen38-fp8-p3-runtime.log
docker logs --since "$REQ_SINCE" "$CONTAINER" >"$REQ_LOG" 2>&1 || true

set +e
python3 - "$MAX_RUNNING" "$START_NS" "$END_NS" "$P3_PROMPT" "$P3_OUTPUT" <<'PY'
import json, pathlib, sys
nr, start_ns, end_ns, expected_prompt, expected_output = map(int, sys.argv[1:])
wall = (end_ns - start_ns) / 1e9
print(f'parallel_wall_sec={wall:.3f}')
ok = True
total_p = total_c = 0
for i in range(nr):
    hp = pathlib.Path(f'/tmp/qwen38-fp8-p3-{i}-http.txt')
    http = hp.read_text().strip() if hp.exists() else ''
    print(f'req{i}_http={http}')
    try:
        d = json.load(open(f'/tmp/qwen38-fp8-p3-{i}-response.json'))
        m = d.get('meta_info', {})
        p = int(m.get('prompt_tokens') or 0)
        c = int(m.get('completion_tokens') or 0)
        print(f'req{i}_prompt_tokens={p}')
        print(f'req{i}_completion_tokens={c}')
        print(f'req{i}_finish_reason={m.get("finish_reason")}')
        total_p += p
        total_c += c
        ok &= http == '200' and p == expected_prompt and c == expected_output
    except Exception as e:
        print(f'req{i}_parse_error={e!r}')
        ok = False
print(f'aggregate_prompt_tokens={total_p}')
print(f'aggregate_completion_tokens={total_c}')
print(f'aggregate_output_tps={total_c / wall:.2f}')
print(f'parallel3_pass={ok}')
if not ok:
    raise SystemExit(1)
PY
PARSE_RC=$?
set -e

if [[ $REQ_RC -ne 0 || $PARSE_RC -ne 0 ]]; then
  echo 'PARALLEL-3 REQUEST FAILURE'
  fail_dump
  exit 1
fi

grep -E 'MTP-CUTOVER-(REQ|PREFILL|DRAFT|EXTEND)|Prefill batch|Decode batch|Scheduler hit an exception|Traceback|CUDA error|out of memory' "$REQ_LOG" | tail -180 || true

echo '=== SINGLE REQUEST 64K PHYSICAL GATE: 65520 + 8 ==='
python3 - <<'PY'
import json
with open('/tmp/qwen38-fp8-64k.json', 'w') as f:
    json.dump({
        'input_ids': [1000] * 65520,
        'sampling_params': {'temperature': 0, 'max_new_tokens': 8, 'ignore_eos': True},
    }, f, separators=(',', ':'))
PY
HTTP64="$(curl --max-time "$REQUEST_TIMEOUT" -sS \
  -o /tmp/qwen38-fp8-64k-response.json \
  -w '%{http_code}' \
  http://127.0.0.1:30000/generate \
  -H 'Content-Type: application/json' \
  --data-binary @/tmp/qwen38-fp8-64k.json || true)"
echo "64k_http=${HTTP64}"
python3 - "$HTTP64" <<'PY'
import json, sys
http = sys.argv[1]
if http != '200':
    raise SystemExit(f'64K HTTP gate failed: {http}')
d = json.load(open('/tmp/qwen38-fp8-64k-response.json'))
m = d.get('meta_info', {})
p = int(m.get('prompt_tokens') or 0)
c = int(m.get('completion_tokens') or 0)
print(f'64k_prompt_tokens={p}')
print(f'64k_completion_tokens={c}')
print(f'64k_finish_reason={m.get("finish_reason")}')
if p != 65520 or c != 8:
    raise SystemExit(f'64K token gate failed: prompt={p} completion={c}')
print('single64k_pass=True')
PY

echo '=== 27K + 1024 MTP ACCEPTANCE / THROUGHPUT GATE ==='
BENCH_LOG="$(mktemp)"
MTP_BENCH_PROMPT_TOKENS=27000 MTP_BENCH_OUTPUT_TOKENS=1024 \
  bash tools/qwen38_mtp_sidecar_benchmark.sh | tee "$BENCH_LOG"
python3 - "$BENCH_LOG" "$MIN_ACCEPT" <<'PY'
from pathlib import Path
import re, sys
s = Path(sys.argv[1]).read_text(errors='replace')
min_accept = float(sys.argv[2])
def one(name, pat, cast=float):
    m = re.search(pat, s, re.M)
    if not m:
        raise SystemExit(f'ERROR: missing {name}')
    return cast(m.group(1))
http = one('HTTP', r'^HTTP=(\d+)$', int)
prompt = one('prompt_tokens', r'^prompt_tokens=(\d+)$', int)
completion = one('completion_tokens', r'^completion_tokens=(\d+)$', int)
accept = one('accept_mean', r'^accept_mean=([0-9.]+)$')
tps = one('output_tps', r'^end_to_end_output_tps=([0-9.]+)$')
print(f'fp8_gate_http={http}')
print(f'fp8_gate_prompt_tokens={prompt}')
print(f'fp8_gate_completion_tokens={completion}')
print(f'fp8_gate_accept_mean={accept:.4f}')
print(f'fp8_gate_output_tps={tps:.2f}')
print(f'fp8_gate_min_accept={min_accept:.2f}')
if http != 200 or prompt != 27000 or completion != 1024:
    raise SystemExit('FP8 FUNCTIONAL GATE FAILED')
if accept < min_accept:
    raise SystemExit(f'FP8 ACCEPTANCE GATE FAILED: {accept:.4f} < {min_accept:.2f}')
print('FP8 ACCEPTANCE GATE PASS')
PY
rm -f "$BENCH_LOG"

echo '=== FINAL CONTAINER STATUS ==='
docker inspect -f 'running={{.State.Running}} status={{.State.Status}} exit={{.State.ExitCode}}' "$CONTAINER"
echo 'FP8 TARGET + FP8 CUDA2 PARALLEL-3 VALIDATION COMPLETE'
