#!/usr/bin/env bash
set -euo pipefail

REPO="${HOME}/projects/sglang-fork"
PATCH_DIR="${HOME}/projects/sglang-patches"
CONTAINER="${MTP_RADIX_CONTAINER:-sglang-qwen38-test}"
IMAGE="${MTP_RADIX_IMAGE:-lmsysorg/sglang:qwen38-27b}"
MODEL="${MTP_RADIX_MODEL:-RadixArk/Qwen3.8-27B-NVFP4}"
PORT="${MTP_RADIX_PORT:-30000}"
MEM_FRACTION_STATIC="${MTP_RADIX_MEM_FRACTION_STATIC:-0.83}"
MAX_RUNNING="${MTP_RADIX_MAX_RUNNING:-3}"
MAX_MAMBA="${MTP_RADIX_MAX_MAMBA:-8}"
PREFIX_TOKENS="${MTP_RADIX_PREFIX_TOKENS:-16384}"
PREFIX_OUTPUT="${MTP_RADIX_PREFIX_OUTPUT:-32}"
MIN_CACHED="${MTP_RADIX_MIN_CACHED:-8192}"
MIN_ACCEPT="${MTP_RADIX_MIN_ACCEPT:-3.0}"
REQUEST_TIMEOUT="${MTP_RADIX_REQUEST_TIMEOUT:-180}"

cd "$REPO"

echo '=== MTP RADIX PREFLIGHT: NO SERVER TOUCH ==='
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
echo 'mtp radix preflight syntax: OK'

echo '=== APPLY PROVEN FP8 SIDECAR PATCH CHAIN ==='
python3 tools/qwen38_mtp_sidecar_cutover.py --commit
python3 tools/qwen38_mtp_cutover_pp_hotfix.py
python3 tools/qwen38_mtp_cutover_mamba_tracking_hotfix.py
python3 tools/qwen38_mtp_cutover_parallel3_hotfix.py --commit
python3 tools/qwen38_mtp_cutover_pool_gate_hotfix.py --commit
python3 tools/qwen38_mtp_cutover_hybrid_fp8_draft_hotfix.py --commit

echo '=== RECREATE RADIX TEST SERVER ==='
docker rm -f "$CONTAINER" >/dev/null 2>&1 || true

docker run -d \
  --name "$CONTAINER" \
  --gpus '"device=0,2,1"' \
  --ipc=host \
  --shm-size 32g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -e SGLANG_MTP_CUTOVER_MIN_POOL_TOKENS=32768 \
  -p "${PORT}:30000" \
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
    --disable-custom-all-reduce \
    --mm-feature-transport cpu \
    --chunked-prefill-size 2048 \
    --max-prefill-tokens 16384 \
    --enable-cache-report \
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
  echo '=== RADIX FAILURE SUMMARY ==='
  docker logs --since "$STARTED" "$CONTAINER" 2>&1 | \
    grep -E 'Tree cache initialized|Unified Radix|MTP-CUTOVER|cached-token|cache hit|Cache|Scheduler hit an exception|chunk state diverged|Traceback|Exception|RuntimeError|ValueError|AssertionError|CUDA error|out of memory|NCCL|Connection closed' | \
    tail -700 || true
  echo '=== RADIX FAILURE TAIL ==='
  docker logs --since "$STARTED" "$CONTAINER" 2>&1 | tail -360 || true
  echo '=== CONTAINER STATUS ==='
  docker inspect -f 'running={{.State.Running}} status={{.State.Status}} exit={{.State.ExitCode}}' "$CONTAINER" 2>/dev/null || true
}

if ! timeout 300 bash -c "
while true; do
  if curl -fsS http://127.0.0.1:${PORT}/model_info >/dev/null 2>&1; then exit 0; fi
  if [ \"\$(docker inspect -f '{{.State.Running}}' ${CONTAINER} 2>/dev/null)\" != \"true\" ]; then exit 2; fi
  sleep 1
done
"; then
  echo 'RADIX SERVER DID NOT BECOME READY'
  fail_dump
  exit 1
fi

echo '=== RADIX STARTUP GATES ==='
START_LOG="$(mktemp)"
trap 'rm -f "$START_LOG" /tmp/qwen38-radix-*.json /tmp/qwen38-radix-*.http /tmp/qwen38-radix-*.log' EXIT
docker logs --since "$STARTED" "$CONTAINER" >"$START_LOG" 2>&1 || true
grep -E 'Tree cache initialized|Init Unified Radix Cache|MTP-CUTOVER|KV Cache|Mamba Cache|max_total_num_tokens|available_gpu_mem' "$START_LOG" | tail -320 || true

if ! grep -Eq 'Tree cache initialized:.*impl=UnifiedRadixCache.*hybrid_ssm=True' "$START_LOG"; then
  echo 'ERROR: target did not initialize UnifiedRadixCache for hybrid SSM'
  fail_dump
  exit 1
fi
if grep -Eq 'disable_radix_cache=True' "$START_LOG"; then
  echo 'ERROR: radix cache is still disabled'
  fail_dump
  exit 1
fi
if ! grep -Eq '\[MTP-CUTOVER-KV\].*draft_kv_dtype=torch\.float8_e4m3fn' "$START_LOG"; then
  echo 'ERROR: CUDA2 draft FP8 gate failed'
  fail_dump
  exit 1
fi
if ! grep -Eq '\[MTP-CUTOVER-MAMBA\].*mamba_slots=8' "$START_LOG"; then
  echo 'ERROR: CUDA2 Mamba8 gate failed'
  fail_dump
  exit 1
fi

TARGET_TOKENS="$(sed -nE 's/.*MTP-CUTOVER-POOL.*target_rank=0 target_tokens=([0-9]+).*/\1/p' "$START_LOG" | tail -1)"
SIDE_TOKENS="$(sed -nE 's/.*MTP-CUTOVER-POOL.*CUDA2 side_tokens=([0-9]+).*/\1/p' "$START_LOG" | tail -1)"
echo "target_tokens=${TARGET_TOKENS:-unknown}"
echo "side_tokens=${SIDE_TOKENS:-unknown}"

make_req() {
  local path="$1" token="$2" n="$3" out="$4"
  python3 - "$path" "$token" "$n" "$out" <<'PY'
import json, sys
path, token, n, out = sys.argv[1], *map(int, sys.argv[2:])
with open(path, 'w') as f:
    json.dump({
        'input_ids': [token] * n,
        'sampling_params': {
            'temperature': 0,
            'max_new_tokens': out,
            'ignore_eos': True,
        },
    }, f, separators=(',', ':'))
PY
}

run_req() {
  local label="$1" body="$2"
  local since http rc=0
  since="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  rm -f "/tmp/qwen38-radix-${label}-response.json" "/tmp/qwen38-radix-${label}.http" "/tmp/qwen38-radix-${label}.log"
  http="$(curl --max-time "$REQUEST_TIMEOUT" -sS \
    -o "/tmp/qwen38-radix-${label}-response.json" \
    -w '%{http_code}' \
    "http://127.0.0.1:${PORT}/generate" \
    -H 'Content-Type: application/json' \
    --data-binary "@${body}" || rc=$?)"
  printf '%s' "$http" >"/tmp/qwen38-radix-${label}.http"
  sleep 1
  docker logs --since "$since" "$CONTAINER" >"/tmp/qwen38-radix-${label}.log" 2>&1 || true
  echo "${label}_curl_rc=${rc}"
  echo "${label}_http=${http}"
  if [[ "$http" == "200" && -s "/tmp/qwen38-radix-${label}-response.json" ]]; then
    python3 - "$label" "/tmp/qwen38-radix-${label}-response.json" <<'PY'
import json, sys
label, path = sys.argv[1:]
d = json.load(open(path))
m = d.get('meta_info', {})
print(f'{label}_prompt_tokens={m.get("prompt_tokens")}')
print(f'{label}_completion_tokens={m.get("completion_tokens")}')
print(f'{label}_cached_tokens_meta={m.get("cached_tokens")}')
PY
  fi
  grep -E 'Prefill batch|Decode batch|MTP-CUTOVER-(REQ|PREFILL|DRAFT|EXTEND)|chunk state diverged|Scheduler hit an exception|Traceback|RuntimeError' "/tmp/qwen38-radix-${label}.log" | tail -260 || true
  [[ $rc -eq 0 && "$http" == "200" ]]
}

REQ=/tmp/qwen38-radix-prefix.json
make_req "$REQ" 1777 "$PREFIX_TOKENS" "$PREFIX_OUTPUT"

echo "=== RADIX COLD REQUEST: ${PREFIX_TOKENS} + ${PREFIX_OUTPUT} ==="
if ! run_req cold "$REQ"; then
  echo 'ERROR: cold radix request failed; prefix-cache reuse was not reached'
  fail_dump
  exit 1
fi

echo "=== RADIX WARM REQUEST: SAME PREFIX ${PREFIX_TOKENS} + ${PREFIX_OUTPUT} ==="
if ! run_req warm "$REQ"; then
  echo 'RADIX WARM REQUEST FAILED'
  echo 'This usually means target prefix reuse advanced seq_len while the independent CUDA2 draft still started from zero.'
  fail_dump
  exit 1
fi

WARM_LOG=/tmp/qwen38-radix-warm.log
read -r WARM_CACHED WARM_ACCEPT <<<"$(python3 - "$WARM_LOG" <<'PY'
from pathlib import Path
import re, sys
s = Path(sys.argv[1]).read_text(errors='replace')
cached = [int(x) for x in re.findall(r'#cached-token:\s*(\d+)', s)]
accept = [float(x) for x in re.findall(r'accept len:\s*([0-9.]+)', s)]
print(max(cached) if cached else 0, sum(accept)/len(accept) if accept else 0.0)
PY
)"
echo "warm_cached_tokens_log=${WARM_CACHED}"
echo "warm_accept_mean_log=${WARM_ACCEPT}"
echo "radix_min_cached=${MIN_CACHED}"
echo "radix_min_accept=${MIN_ACCEPT}"
python3 - "$WARM_CACHED" "$MIN_CACHED" "$WARM_ACCEPT" "$MIN_ACCEPT" <<'PY'
import sys
cached, min_cached = map(int, sys.argv[1:3])
accept, min_accept = map(float, sys.argv[3:5])
if cached < min_cached:
    raise SystemExit(f'RADIX CACHE HIT GATE FAILED: {cached} < {min_cached}')
if accept < min_accept:
    raise SystemExit(f'RADIX MTP ACCEPTANCE GATE FAILED: {accept:.4f} < {min_accept:.2f}')
print('RADIX PREFIX + MTP GATE PASS')
PY

echo '=== RADIX SINGLE REQUEST 64K GATE ==='
make_req /tmp/qwen38-radix-64k.json 1888 65520 8
if ! run_req single64k /tmp/qwen38-radix-64k.json; then
  echo 'RADIX 64K GATE FAILED'
  fail_dump
  exit 1
fi
python3 - /tmp/qwen38-radix-single64k-response.json <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
m = d.get('meta_info', {})
p = int(m.get('prompt_tokens') or 0)
c = int(m.get('completion_tokens') or 0)
print(f'radix_64k_prompt_tokens={p}')
print(f'radix_64k_completion_tokens={c}')
if p != 65520 or c != 8:
    raise SystemExit(f'RADIX 64K TOKEN GATE FAILED: prompt={p} completion={c}')
print('radix_single64k_pass=True')
PY

echo '=== RADIX FINAL CONTAINER STATUS ==='
docker inspect -f 'running={{.State.Running}} status={{.State.Status}} exit={{.State.ExitCode}}' "$CONTAINER"
echo 'MTP RADIX COMPATIBILITY VALIDATION COMPLETE'
