#!/usr/bin/env bash
set -euo pipefail

CONTAINER="${MTP_CUTOVER_CONTAINER:-sglang-qwen38-test}"
PORT="${MTP_CUTOVER_PORT:-30000}"
PROMPT_TOKENS="${MTP_BENCH_PROMPT_TOKENS:-27000}"
OUTPUT_TOKENS="${MTP_BENCH_OUTPUT_TOKENS:-1024}"
TOKEN_ID="${MTP_BENCH_TOKEN_ID:-1000}"

if [[ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER" 2>/dev/null || true)" != "true" ]]; then
  echo "ERROR: container $CONTAINER is not running" >&2
  exit 1
fi

curl -fsS "http://127.0.0.1:${PORT}/model_info" >/dev/null

PAYLOAD=/tmp/qwen38-mtp-sidecar-bench.json
RESPONSE=/tmp/qwen38-mtp-sidecar-bench-response.json
LOG=/tmp/qwen38-mtp-sidecar-bench.log

python3 - "$PAYLOAD" "$PROMPT_TOKENS" "$OUTPUT_TOKENS" "$TOKEN_ID" <<'PY'
import json, sys
path, prompt_tokens, output_tokens, token_id = sys.argv[1:]
payload = {
    "input_ids": [int(token_id)] * int(prompt_tokens),
    "sampling_params": {
        "temperature": 0,
        "max_new_tokens": int(output_tokens),
        "ignore_eos": True,
    },
}
with open(path, "w") as f:
    json.dump(payload, f, separators=(",", ":"))
PY

SINCE="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
START_NS="$(date +%s%N)"
HTTP="$(curl -sS -o "$RESPONSE" -w '%{http_code}' \
  "http://127.0.0.1:${PORT}/generate" \
  -H 'Content-Type: application/json' \
  --data-binary @"$PAYLOAD" || true)"
END_NS="$(date +%s%N)"

sleep 1
docker logs --since "$SINCE" "$CONTAINER" >"$LOG" 2>&1 || true

echo "HTTP=${HTTP}"
if [[ "$HTTP" != "200" ]]; then
  echo '=== RESPONSE ==='
  cat "$RESPONSE" 2>/dev/null || true
  echo
  echo '=== CUTOVER FAILURE LOG ==='
  grep -A100 -B20 -E \
    'MTP-CUTOVER|Scheduler hit an exception|Traceback|RuntimeError|AssertionError|CUDA error|illegal memory|out of memory' \
    "$LOG" | tail -320 || true
  exit 1
fi

python3 - "$RESPONSE" "$LOG" "$START_NS" "$END_NS" "$PROMPT_TOKENS" "$OUTPUT_TOKENS" <<'PY'
import json, re, statistics, sys
response_path, log_path, start_ns, end_ns, requested_prompt, requested_output = sys.argv[1:]
d = json.load(open(response_path))
meta = d.get("meta_info", {})
wall = (int(end_ns) - int(start_ns)) / 1e9
prompt = int(meta.get("prompt_tokens") or requested_prompt)
completion = int(meta.get("completion_tokens") or 0)

log = open(log_path, errors="replace").read()
accepts = [int(x) for x in re.findall(r"MTP-CUTOVER-EXTEND\].*?accept=(\d+)", log)]
proposals = re.findall(r"MTP-CUTOVER-DRAFT\].*?proposal=(\[\[.*?\]\])", log)

print(f"prompt_tokens={prompt}")
print(f"completion_tokens={completion}")
print(f"finish_reason={meta.get('finish_reason')}")
print(f"wall_sec={wall:.3f}")
print(f"end_to_end_total_tps={(prompt + completion) / wall:.2f}")
print(f"end_to_end_output_tps={completion / wall:.2f}")
if accepts:
    print(f"verify_iterations={len(accepts)}")
    print(f"accept_mean={statistics.mean(accepts):.4f}")
    print(f"accept_median={statistics.median(accepts):.2f}")
    print(f"accept_min={min(accepts)}")
    print(f"accept_max={max(accepts)}")
    print(f"accepted_tokens_total={sum(accepts)}")
    print(f"effective_tokens_per_verify={sum(accepts)/len(accepts):.4f}")
else:
    print("verify_iterations=0")
print(f"draft_log_entries={len(proposals)}")

# SGLang's final decode log is useful because it excludes most prefill cost.
decode = re.findall(
    r"Decode batch.*?gen throughput \(token/s\): ([0-9.]+).*?accept length: ([0-9.]+)",
    log,
)
if decode:
    print(f"last_decode_gen_tps={float(decode[-1][0]):.2f}")
    print(f"last_decode_accept_length={float(decode[-1][1]):.2f}")
PY

echo '=== CUTOVER BENCH LOG TAIL ==='
grep -E \
  'MTP-CUTOVER-(REQ|PREFILL|DRAFT|EXTEND)|Prefill batch|Decode batch|Scheduler hit an exception|CUDA error|illegal memory|Traceback' \
  "$LOG" | tail -120 || true

echo '=== CONTAINER STATUS ==='
docker inspect -f 'running={{.State.Running}} status={{.State.Status}} exit={{.State.ExitCode}}' "$CONTAINER"
