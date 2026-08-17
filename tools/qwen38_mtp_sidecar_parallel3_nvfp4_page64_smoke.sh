#!/usr/bin/env bash
set -euo pipefail

REPO="${HOME}/projects/sglang-fork"
BASE="${REPO}/tools/qwen38_mtp_sidecar_parallel3_nvfp4_smoke.sh"
TMP="$(mktemp)"
trap 'rm -f "$TMP"' EXIT

cd "$REPO"

# Reuse the proven NVFP4 target+draft smoke, but swap only the sidecar paging
# hotfix/gates.  Keeping this as a wrapper avoids duplicating a long test script.
sed \
  -e 's/qwen38_mtp_cutover_sidecar_page1_hotfix.py/qwen38_mtp_cutover_sidecar_page64_hotfix.py/g' \
  -e 's/PAGE_SIZE=1/PAGE_SIZE=64/g' \
  -e 's/page_size=1/page_size=64/g' \
  -e 's/draft_page_size=1/draft_page_size=64/g' \
  -e 's/allocator=TokenToKVPoolAllocator/allocator=PagedTokenToKVPoolAllocator/g' \
  "$BASE" > "$TMP"

# Guard against accidental stale page1 expectations before touching the server.
if grep -Eq 'page_size=1|draft_page_size=1|sidecar_page1_hotfix' "$TMP"; then
  echo 'ERROR: generated page64 smoke still contains page1 expectations'
  grep -nE 'page_size=1|draft_page_size=1|sidecar_page1_hotfix' "$TMP" || true
  exit 1
fi

bash -n "$TMP"
echo 'page64 smoke wrapper syntax: OK'
exec bash "$TMP"
