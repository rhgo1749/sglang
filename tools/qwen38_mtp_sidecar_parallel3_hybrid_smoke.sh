#!/usr/bin/env bash
set -euo pipefail

REPO="${HOME}/projects/sglang-fork"
BASE="${REPO}/tools/qwen38_mtp_sidecar_parallel3_nvfp4_smoke.sh"
TMP="$(mktemp)"
trap 'rm -f "$TMP"' EXIT

cd "$REPO"
python3 -m py_compile tools/qwen38_mtp_cutover_hybrid_fp8_draft_hotfix.py

# Do not let a failed curl parse response JSON from a previous successful run.
rm -f /tmp/qwen38-nvfp4-p3-*.json \
      /tmp/qwen38-nvfp4-p3-*-http.txt \
      /tmp/qwen38-nvfp4-p3-runtime.log 2>/dev/null || true

# Reuse the mature parallel-3 harness, but deterministically rewrite the
# experimental NVFP4-draft gates to the production hybrid gates.  Python exact
# replacements are used here instead of a stack of sed regexes so backslashes in
# grep patterns cannot silently miss a transform.
python3 - "$BASE" "$TMP" <<'PY'
from pathlib import Path
import sys

base, out = map(Path, sys.argv[1:])
s = base.read_text()

# The hybrid hotfix supersedes both the NVFP4-draft A/B switch and the old page1
# hotfix.  It owns dtype/backend/page migration as one atomic operation.
s = s.replace(
    "tools/qwen38_mtp_cutover_nvfp4_target_hotfix.py",
    "tools/qwen38_mtp_cutover_hybrid_fp8_draft_hotfix.py",
)
s = s.replace("  tools/qwen38_mtp_cutover_sidecar_page1_hotfix.py\n", "")
s = s.replace(
    "python3 tools/qwen38_mtp_cutover_sidecar_page1_hotfix.py --commit\n", ""
)
s = s.replace("=== HOTFIX CUDA2 SIDECAR PAGE_SIZE=1 ===\n", "")

repl = {
    "NVFP4 TARGET+DRAFT ABI": "NVFP4 TARGET + FP8 DRAFT ABI",
    "CUDA2 draft: NVFP4 KV (FlashInfer extend + TRTLLM-MHA multi-step decode, private page_size=1)":
        "CUDA2 draft: FP8 E4M3 KV (FlashInfer multi-step decode + extend, private page_size=1)",
    "HOTFIX NVFP4 TARGET / NVFP4 CUDA2 DRAFT": "HOTFIX NVFP4 TARGET / FP8 CUDA2 DRAFT",
    "RECREATE SERVER: NVFP4 TARGET KV, NVFP4 DRAFT KV": "RECREATE SERVER: NVFP4 TARGET KV, FP8 DRAFT KV",
    "target=nvfp4 CUDA2 draft=nvfp4": "target=nvfp4 CUDA2 draft=fp8_e4m3",
    r"draft_kv_dtype=torch\.float4_e2m1fn_x2.*draft_kv_tag=nvfp4":
        r"draft_kv_dtype=torch\.float8_e4m3fn.*draft_kv_tag=fp8_e4m3",
    "CUDA2 private NVFP4 draft-KV override did not activate":
        "CUDA2 private FP8 draft-KV override did not activate",
    "CUDA2 draft runner did not resolve NVFP4 KV":
        "CUDA2 draft runner did not resolve FP8 E4M3 KV",
    r"TRTLLMHAAttnMultiStepDraftBackend.*extend=FlashInferAttnBackend":
        r"FlashInferMultiStepDraftBackend.*extend=FlashInferAttnBackend",
    "CUDA2 draft did not resolve TRTLLM-MHA decode + FlashInfer extend split":
        "CUDA2 draft did not resolve FlashInfer decode + extend",
    "draft_kv=nvfp4": "draft_kv=fp8_e4m3",
    "MTP NVFP4-TARGET / NVFP4-DRAFT PARALLEL-3 SMOKE COMPLETE":
        "MTP NVFP4-TARGET / FP8-DRAFT PARALLEL-3 SMOKE COMPLETE",
}
for old, new in repl.items():
    if old not in s:
        raise SystemExit(f"ERROR: hybrid smoke transform point missing: {old}")
    s = s.replace(old, new)

for forbidden in (
    "draft=nvfp4",
    "draft_kv=nvfp4",
    "draft_kv_tag=nvfp4",
    "TRTLLMHAAttnMultiStepDraftBackend.*extend=FlashInferAttnBackend",
    "qwen38_mtp_cutover_sidecar_page1_hotfix.py",
):
    if forbidden in s:
        raise SystemExit(f"ERROR: stale experimental gate remains: {forbidden}")

out.write_text(s)
PY

bash -n "$TMP"
echo 'hybrid smoke wrapper syntax: OK'
exec bash "$TMP"
