#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:8080}"
MODEL="${MODEL:-qwen3.8-27b-NVFP4-Q5K-MTP}"
SHORT_PROMPT_TOKENS="${SHORT_PROMPT_TOKENS:-4096}"
SHORT_DECODE_TOKENS="${SHORT_DECODE_TOKENS:-1024}"
LONG_PROMPT_TOKENS="${LONG_PROMPT_TOKENS:-262000}"
LONG_DECODE_TOKENS="${LONG_DECODE_TOKENS:-8}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-1800}"
ROOT="${ROOT:-/tmp/llamacpp-qwen38-parallel2-bench}"
RUN_SHORT="${RUN_SHORT:-1}"
RUN_LONG="${RUN_LONG:-1}"

mkdir -p "$ROOT"
health_url="${BASE_URL%/}/health"
completion_url="${BASE_URL%/}/completion"

echo '============================================================'
echo ' llama.cpp Qwen3.8 parallel-2 wall-clock benchmark'
echo " base_url=${BASE_URL}"
echo " model=${MODEL}"
echo " short=${SHORT_PROMPT_TOKENS}+${SHORT_DECODE_TOKENS} x2 concurrent"
echo " long=${LONG_PROMPT_TOKENS}+${LONG_DECODE_TOKENS} x2 concurrent"
echo ' cache_prompt=false / temperature=0 / ignore_eos=true'
echo '============================================================'

if ! curl -fsS --max-time 10 "$health_url" >/dev/null; then
  echo "ERROR: llama.cpp health check failed: ${health_url}" >&2
  exit 10
fi

# Warm-load the selected router preset so model loading is excluded from the timed cases.
python3 - "$completion_url" "$MODEL" "$REQUEST_TIMEOUT" <<'PY'
import json,sys,urllib.request
url,model,timeout=sys.argv[1],sys.argv[2],float(sys.argv[3])
payload={
    "model":model,
    "prompt":[1199]*256,
    "n_predict":8,
    "temperature":0,
    "ignore_eos":True,
    "cache_prompt":False,
    "stream":False,
}
req=urllib.request.Request(url,data=json.dumps(payload).encode(),headers={"Content-Type":"application/json"},method="POST")
with urllib.request.urlopen(req,timeout=timeout) as r:
    raw=r.read()
    if r.status != 200:
        raise SystemExit(f"warm-load HTTP {r.status}: {raw[:500]!r}")
print('WARM_LOAD_PASS=True')
PY

python3 - "$completion_url" "$MODEL" "$ROOT" "$SHORT_PROMPT_TOKENS" "$SHORT_DECODE_TOKENS" "$LONG_PROMPT_TOKENS" "$LONG_DECODE_TOKENS" "$REQUEST_TIMEOUT" "$RUN_SHORT" "$RUN_LONG" <<'PY'
import concurrent.futures
import json
import pathlib
import sys
import time
import urllib.error
import urllib.request

(
    url, model, root_s, short_prompt_s, short_decode_s,
    long_prompt_s, long_decode_s, timeout_s, run_short_s, run_long_s,
) = sys.argv[1:]
root=pathlib.Path(root_s)
short_prompt=int(short_prompt_s); short_decode=int(short_decode_s)
long_prompt=int(long_prompt_s); long_decode=int(long_decode_s)
timeout=float(timeout_s)
run_short=run_short_s == '1'; run_long=run_long_s == '1'

def one(label, i, prompt_tokens, decode_tokens):
    payload={
        "model": model,
        "prompt": [1200+i]*prompt_tokens,
        "n_predict": decode_tokens,
        "temperature": 0,
        "ignore_eos": True,
        "cache_prompt": False,
        "stream": False,
    }
    body=json.dumps(payload,separators=(',',':')).encode()
    req=urllib.request.Request(url,data=body,headers={"Content-Type":"application/json"},method='POST')
    t0=time.perf_counter()
    try:
        with urllib.request.urlopen(req,timeout=timeout) as r:
            raw=r.read(); status=r.status
    except urllib.error.HTTPError as e:
        raw=e.read(); status=e.code
    elapsed=time.perf_counter()-t0
    (root/f'{label}-req{i}.json').write_bytes(raw)
    try:
        d=json.loads(raw)
    except Exception as e:
        return {"i":i,"status":status,"elapsed":elapsed,"error":f'invalid JSON {e!r}',"raw":repr(raw[:500])}
    if status != 200:
        return {"i":i,"status":status,"elapsed":elapsed,"error":d}
    t=d.get('timings') or {}
    return {
        "i":i,"status":status,"elapsed":elapsed,
        "prompt_n":int(t.get('prompt_n') or d.get('tokens_evaluated') or 0),
        "cache_n":int(t.get('cache_n') or d.get('tokens_cached') or 0),
        "predicted_n":int(t.get('predicted_n') or d.get('tokens_predicted') or 0),
        "prompt_ms":t.get('prompt_ms'),
        "prompt_tps":t.get('prompt_per_second'),
        "predicted_ms":t.get('predicted_ms'),
        "predicted_tps":t.get('predicted_per_second'),
        "truncated":bool(d.get('truncated',False)),
    }

def run_case(label,prompt_tokens,decode_tokens):
    print(f'=== {label.upper()} PARALLEL-2 ===')
    t0=time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        futs=[ex.submit(one,label,i,prompt_tokens,decode_tokens) for i in range(2)]
        rows=[f.result() for f in futs]
    total=time.perf_counter()-t0
    rows.sort(key=lambda x:x['i'])
    ok=True
    for r in rows:
        i=r['i']
        print(f'{label}_req{i}_http={r.get("status")}')
        print(f'{label}_req{i}_wall_seconds={r.get("elapsed",0):.3f}')
        if 'error' in r:
            print(f'{label}_req{i}_error={r["error"]}')
            ok=False
            continue
        for k in ('prompt_n','cache_n','predicted_n'):
            print(f'{label}_req{i}_{k}={r[k]}')
        for k in ('prompt_ms','prompt_tps','predicted_ms','predicted_tps'):
            v=r.get(k)
            if isinstance(v,(int,float)):
                print(f'{label}_req{i}_{k}={v:.3f}')
        print(f'{label}_req{i}_truncated={r["truncated"]}')
        exact=(r['prompt_n']==prompt_tokens and r['predicted_n']==decode_tokens and not r['truncated'])
        print(f'{label}_req{i}_exact_token_pass={exact}')
        ok &= exact
    print(f'{label}_parallel2_wall_seconds={total:.3f}')
    print(f'{label}_parallel2_effective_completion_tps={(2*decode_tokens)/total:.3f}')
    print(f'{label}_parallel2_pass={ok}')
    return total,ok

short=None; long=None
if run_short:
    short=run_case('short',short_prompt,short_decode)
if run_long:
    long=run_case('long',long_prompt,long_decode)

print('=== LLAMA.CPP PARALLEL-2 BENCH SUMMARY ===')
if short is not None:
    total,ok=short
    print(f'llamacpp_parallel2_short_seconds={total:.3f}')
    print('llamacpp_serial1_short_reference_seconds=16.061')
    print('sglang_parallel2_short_reference_seconds=98.922')
    print(f'parallel2_vs_llamacpp_serial2_speedup={16.061/total:.3f}')
    print(f'parallel2_vs_sglang_wall_ratio={total/98.922:.3f}')
    print(f'llamacpp_parallel2_short_pass={ok}')
if long is not None:
    total,ok=long
    print(f'llamacpp_parallel2_long_seconds={total:.3f}')
    print('llamacpp_serial2_long_reference_seconds=402.423')
    print(f'parallel2_vs_llamacpp_serial2_long_speedup={402.423/total:.3f}')
    print(f'llamacpp_parallel2_long_pass={ok}')
print('LLAMACPP_QWEN38_PARALLEL2_BENCH=DONE')
PY
