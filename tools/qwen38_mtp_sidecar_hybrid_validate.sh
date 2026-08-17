#!/usr/bin/env bash
set -euo pipefail

REPO="${HOME}/projects/sglang-fork"
MIN_ACCEPT="${MTP_HYBRID_MIN_ACCEPT:-2.0}"
BENCH_LOG="$(mktemp)"
trap 'rm -f "$BENCH_LOG"' EXIT

cd "$REPO"

echo '=== HYBRID PREFLIGHT: SYNTAX ONLY, NO SERVER TOUCH ==='
PY_FILES=(
  tools/qwen38_mtp_sidecar_cutover.py
  tools/qwen38_mtp_cutover_pp_hotfix.py
  tools/qwen38_mtp_cutover_mamba_tracking_hotfix.py
  tools/qwen38_mtp_cutover_parallel3_hotfix.py
  tools/qwen38_mtp_cutover_pool_gate_hotfix.py
  tools/qwen38_mtp_cutover_hybrid_fp8_draft_hotfix.py
)
for f in "${PY_FILES[@]}"; do
  python3 -m py_compile "$f"
done
bash -n tools/qwen38_mtp_sidecar_parallel3_hybrid_smoke.sh
bash -n tools/qwen38_mtp_sidecar_benchmark.sh
echo 'hybrid preflight syntax: OK'

bash tools/qwen38_mtp_sidecar_parallel3_hybrid_smoke.sh

echo '=== HYBRID 27K+1024 ACCEPTANCE BENCH ==='
bash tools/qwen38_mtp_sidecar_benchmark.sh | tee "$BENCH_LOG"

python3 - "$BENCH_LOG" "$MIN_ACCEPT" <<'PY'
from pathlib import Path
import re, sys

p = Path(sys.argv[1])
min_accept = float(sys.argv[2])
s = p.read_text(errors="replace")

def one(name, pattern, cast=float):
    m = re.search(pattern, s, re.M)
    if not m:
        raise SystemExit(f"ERROR: benchmark did not report {name}")
    return cast(m.group(1))

http = one("HTTP", r"^HTTP=(\d+)$", int)
prompt = one("prompt_tokens", r"^prompt_tokens=(\d+)$", int)
completion = one("completion_tokens", r"^completion_tokens=(\d+)$", int)
accept = one("accept_mean", r"^accept_mean=([0-9.]+)$")
out_tps = one("end_to_end_output_tps", r"^end_to_end_output_tps=([0-9.]+)$")

print(f"hybrid_gate_http={http}")
print(f"hybrid_gate_prompt_tokens={prompt}")
print(f"hybrid_gate_completion_tokens={completion}")
print(f"hybrid_gate_accept_mean={accept:.4f}")
print(f"hybrid_gate_output_tps={out_tps:.2f}")
print(f"hybrid_gate_min_accept={min_accept:.2f}")

if http != 200 or prompt != 27000 or completion != 1024:
    raise SystemExit("HYBRID FUNCTIONAL GATE FAILED")
if accept < min_accept:
    raise SystemExit(
        f"HYBRID ACCEPTANCE GATE FAILED: {accept:.4f} < {min_accept:.2f}"
    )
print("HYBRID ACCEPTANCE GATE PASS")
PY
