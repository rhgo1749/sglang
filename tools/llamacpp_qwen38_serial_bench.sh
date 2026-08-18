#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:8080}"
SHORT_PROMPT_TOKENS="${SHORT_PROMPT_TOKENS:-4096}"
SHORT_DECODE_TOKENS="${SHORT_DECODE_TOKENS:-1024}"
LONG_PROMPT_TOKENS="${LONG_PROMPT_TOKENS:-262000}"
LONG_DECODE_TOKENS="${LONG_DECODE_TOKENS:-8}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-1800}"
ROOT="${ROOT:-/tmp/llamacpp-qwen38-serial-bench}"
RUN_SHORT="${RUN_SHORT:-1}"
RUN_LONG="${RUN_LONG:-1}"

mkdir -p "$ROOT"

health_url="${BASE_URL%/}/health"
completion_url="${BASE_URL%/}/completion"

echo '============================================================'
echo ' llama.cpp Qwen3.8 serial wall-clock benchmark'
echo " base_url=${BASE_URL}"
echo " short=${SHORT_PROMPT_TOKENS}+${SHORT_DECODE_TOKENS} x2 serial"
echo " long=${LONG_PROMPT_TOKENS}+${LONG_DECODE_TOKENS} x2 serial"
echo ' cache_prompt=false / temperature=0 / ignore_eos=true'
echo ' Native /completion with numeric token arrays: no tokenizer/template drift.'
echo '============================================================'

if ! curl -fsS --max-time 10 "$health_url" >/dev/null; then
  echo "ERROR: llama.cpp health check failed: ${health_url}" >&2
  echo 'If :8080 is only a router and does not expose llama.cpp native endpoints,' >&2
  echo 'set BASE_URL to the actual llama-server backend URL.' >&2
  exit 10
fi

python3 - "$completion_url" "$ROOT" "$SHORT_PROMPT_TOKENS" "$SHORT_DECODE_TOKENS" "$LONG_PROMPT_TOKENS" "$LONG_DECODE_TOKENS" "$REQUEST_TIMEOUT" "$RUN_SHORT" "$RUN_LONG" <<'PY'
import json
import pathlib
import sys
import time
import urllib.error
import urllib.request

(
    url,
    root_s,
    short_prompt_s,
    short_decode_s,
    long_prompt_s,
    long_decode_s,
    timeout_s,
    run_short_s,
    run_long_s,
) = sys.argv[1:]
root = pathlib.Path(root_s)
short_prompt = int(short_prompt_s)
short_decode = int(short_decode_s)
long_prompt = int(long_prompt_s)
long_decode = int(long_decode_s)
timeout = float(timeout_s)
run_short = run_short_s == "1"
run_long = run_long_s == "1"


def request_once(label: str, req_index: int, prompt_tokens: int, decode_tokens: int):
    # Match the SGLang synthetic stress pattern: two distinct constant-token prompts.
    token_id = 1200 + req_index
    payload = {
        "prompt": [token_id] * prompt_tokens,
        "n_predict": decode_tokens,
        "temperature": 0,
        "ignore_eos": True,
        "cache_prompt": False,
        "stream": False,
    }
    body = json.dumps(payload, separators=(",", ":")).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read()
            status = r.status
    except urllib.error.HTTPError as e:
        raw = e.read()
        status = e.code
    elapsed = time.perf_counter() - t0
    out = root / f"{label}-req{req_index}.json"
    out.write_bytes(raw)
    try:
        data = json.loads(raw)
    except Exception as e:
        raise SystemExit(f"{label} req{req_index}: HTTP {status}, invalid JSON: {e!r}; body={raw[:500]!r}")
    if status != 200:
        raise SystemExit(f"{label} req{req_index}: HTTP {status}: {data}")

    timings = data.get("timings") or {}
    prompt_n = int(timings.get("prompt_n") or data.get("tokens_evaluated") or 0)
    cache_n = int(timings.get("cache_n") or data.get("tokens_cached") or 0)
    predicted_n = int(timings.get("predicted_n") or data.get("tokens_predicted") or 0)
    # With cache_prompt=false we expect the entire synthetic prompt to be evaluated.
    # Some builds report prompt_n excluding a compulsory/internal token; native numeric
    # token arrays should normally be exact, so print rather than silently normalize.
    print(f"{label}_req{req_index}_http={status}")
    print(f"{label}_req{req_index}_wall_seconds={elapsed:.3f}")
    print(f"{label}_req{req_index}_prompt_n={prompt_n}")
    print(f"{label}_req{req_index}_cache_n={cache_n}")
    print(f"{label}_req{req_index}_predicted_n={predicted_n}")
    for src, dst in [
        ("prompt_ms", "prompt_ms"),
        ("prompt_per_second", "prompt_tps"),
        ("predicted_ms", "predicted_ms"),
        ("predicted_per_second", "predicted_tps"),
    ]:
        if src in timings:
            v = timings[src]
            if isinstance(v, (int, float)):
                print(f"{label}_req{req_index}_{dst}={v:.3f}")
            else:
                print(f"{label}_req{req_index}_{dst}={v}")
    print(f"{label}_req{req_index}_truncated={bool(data.get('truncated', False))}")
    ok = (
        prompt_n == prompt_tokens
        and predicted_n == decode_tokens
        and not bool(data.get("truncated", False))
    )
    print(f"{label}_req{req_index}_exact_token_pass={ok}")
    if not ok:
        raise SystemExit(
            f"{label} req{req_index}: exact-token check failed: "
            f"prompt_n={prompt_n}/{prompt_tokens}, predicted_n={predicted_n}/{decode_tokens}, "
            f"truncated={data.get('truncated', False)}"
        )
    return elapsed, timings


def run_case(label: str, prompt_tokens: int, decode_tokens: int):
    print(f"=== {label.upper()} SERIAL-2 ===")
    total_t0 = time.perf_counter()
    rows = []
    for i in range(2):
        rows.append(request_once(label, i, prompt_tokens, decode_tokens))
    total = time.perf_counter() - total_t0
    wall_sum = sum(x[0] for x in rows)
    print(f"{label}_serial2_wall_seconds={total:.3f}")
    print(f"{label}_serial2_request_wall_sum_seconds={wall_sum:.3f}")
    print(f"{label}_serial2_effective_completion_tps={(2*decode_tokens)/total:.3f}")
    prompt_ms = sum(float(x[1].get("prompt_ms") or 0.0) for x in rows)
    pred_ms = sum(float(x[1].get("predicted_ms") or 0.0) for x in rows)
    if prompt_ms:
        print(f"{label}_serial2_engine_prompt_tps={(2*prompt_tokens)/(prompt_ms/1000):.3f}")
    if pred_ms:
        print(f"{label}_serial2_engine_generation_tps={(2*decode_tokens)/(pred_ms/1000):.3f}")
    print(f"{label}_serial2_pass=True")
    return total

short_total = None
long_total = None
if run_short:
    short_total = run_case("short", short_prompt, short_decode)
if run_long:
    long_total = run_case("long", long_prompt, long_decode)

print("=== LLAMA.CPP BENCH SUMMARY ===")
if short_total is not None:
    sglang_short = 98.922
    print(f"llamacpp_short_serial2_seconds={short_total:.3f}")
    print(f"sglang_short_parallel2_reference_seconds={sglang_short:.3f}")
    print(f"llamacpp_short_vs_sglang_wall_ratio={short_total/sglang_short:.3f}")
    print(f"llamacpp_short_faster_than_sglang={short_total < sglang_short}")
if long_total is not None:
    print(f"llamacpp_long_serial2_seconds={long_total:.3f}")
print("LLAMACPP_QWEN38_SERIAL_BENCH=PASS")
PY
