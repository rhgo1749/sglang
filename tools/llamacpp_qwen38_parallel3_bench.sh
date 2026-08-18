#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:8080}"
MODEL="${MODEL:-qwen3.8-27b-NVFP4-Q5K-MTP}"
PROMPT_TOKENS="${PROMPT_TOKENS:-4096}"
DECODE_TOKENS="${DECODE_TOKENS:-1024}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-1800}"
ROOT="${ROOT:-/tmp/llamacpp-qwen38-parallel3-bench}"

mkdir -p "$ROOT"
health_url="${BASE_URL%/}/health"
completion_url="${BASE_URL%/}/completion"

echo '============================================================'
echo ' llama.cpp Qwen3.8 parallel-3 wall-clock benchmark'
echo " base_url=${BASE_URL}"
echo " model=${MODEL}"
echo " workload=${PROMPT_TOKENS}+${DECODE_TOKENS} x3 concurrent"
echo ' cache_prompt=false / temperature=0 / ignore_eos=true'
echo '============================================================'

if ! curl -fsS --max-time 10 "$health_url" >/dev/null; then
  echo "ERROR: llama.cpp health check failed: ${health_url}" >&2
  exit 10
fi

# Warm-load selected router preset before the timed run.
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

python3 - "$completion_url" "$MODEL" "$ROOT" "$PROMPT_TOKENS" "$DECODE_TOKENS" "$REQUEST_TIMEOUT" <<'PY'
import concurrent.futures
import json
import pathlib
import sys
import time
import urllib.error
import urllib.request

url,model,root_s,prompt_s,decode_s,timeout_s=sys.argv[1:]
root=pathlib.Path(root_s)
prompt_tokens=int(prompt_s); decode_tokens=int(decode_s); timeout=float(timeout_s)

def one(i):
    payload={
        "model":model,
        "prompt":[1200+i]*prompt_tokens,
        "n_predict":decode_tokens,
        "temperature":0,
        "ignore_eos":True,
        "cache_prompt":False,
        "stream":False,
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
    (root/f'req{i}.json').write_bytes(raw)
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

print('=== PARALLEL-3 ===')
t0=time.perf_counter()
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
    rows=[f.result() for f in [ex.submit(one,i) for i in range(3)]]
total=time.perf_counter()-t0
rows.sort(key=lambda x:x['i'])
ok=True
for r in rows:
    i=r['i']
    print(f'req{i}_http={r.get("status")}')
    print(f'req{i}_wall_seconds={r.get("elapsed",0):.3f}')
    if 'error' in r:
        print(f'req{i}_error={r["error"]}')
        ok=False
        continue
    for k in ('prompt_n','cache_n','predicted_n'):
        print(f'req{i}_{k}={r[k]}')
    for k in ('prompt_ms','prompt_tps','predicted_ms','predicted_tps'):
        v=r.get(k)
        if isinstance(v,(int,float)):
            print(f'req{i}_{k}={v:.3f}')
    print(f'req{i}_truncated={r["truncated"]}')
    exact=(r['prompt_n']==prompt_tokens and r['predicted_n']==decode_tokens and not r['truncated'])
    print(f'req{i}_exact_token_pass={exact}')
    ok &= exact

print(f'parallel3_wall_seconds={total:.3f}')
print(f'parallel3_effective_completion_tps={(3*decode_tokens)/total:.3f}')
print(f'parallel3_pass={ok}')
print('=== REFERENCES ===')
print('dual5070_parallel1_serial2_4096x1024_seconds=16.061')
print('threegpu_parallel3_parallel2_4096x1024_seconds=12.887')
print(f'parallel3_vs_parallel2_wall_ratio={total/12.887:.3f}')
print(f'parallel3_vs_parallel2_total_completion_gain={(3*decode_tokens/total)/(2*decode_tokens/12.887):.3f}')
print('LLAMACPP_QWEN38_PARALLEL3_BENCH=DONE')
raise SystemExit(0 if ok else 1)
PY
