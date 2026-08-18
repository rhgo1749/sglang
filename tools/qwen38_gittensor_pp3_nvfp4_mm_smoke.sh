#!/usr/bin/env bash
set -euo pipefail

PORT="${PORT:-30001}"
MODEL="${MODEL:-gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090}"
CONTAINER="${CONTAINER:-sglang-qwen38-gittensor-pp3}"
ROOT="${ROOT:-/tmp/qwen38-mm-smoke}"

mkdir -p "$ROOT"

if ! curl -fsS --max-time 5 "http://127.0.0.1:${PORT}/model_info" >/dev/null; then
  echo "MM_SMOKE_SERVER_READY=False"
  echo "Start the validated serve preset first." >&2
  exit 2
fi

echo "MM_SMOKE_SERVER_READY=True"

# Generate a self-contained 64x64 solid-red RGB PNG using only stdlib, then
# embed it as a data URL so the smoke test has no network dependency.
python3 - "$ROOT/request.json" "$MODEL" <<'PY'
import base64, binascii, json, struct, sys, zlib

out, model = sys.argv[1], sys.argv[2]
w = h = 64
raw = b"".join(b"\x00" + bytes((255, 0, 0)) * w for _ in range(h))

def chunk(kind: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + kind
        + data
        + struct.pack(">I", binascii.crc32(kind + data) & 0xFFFFFFFF)
    )

png = (
    b"\x89PNG\r\n\x1a\n"
    + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
    + chunk(b"IDAT", zlib.compress(raw, 9))
    + chunk(b"IEND", b"")
)
url = "data:image/png;base64," + base64.b64encode(png).decode("ascii")
req = {
    "model": model,
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": url}},
                {
                    "type": "text",
                    "text": "What is the dominant color of this image? Reply with exactly one English color word.",
                },
            ],
        }
    ],
    "temperature": 0,
    # Qwen reasoning can consume the first few dozen completion tokens before
    # publishing final content. 32 tokens was too small for a semantic smoke and
    # could end with finish_reason=length while the vision path itself was fine.
    "max_tokens": 128,
}
with open(out, "w") as f:
    json.dump(req, f, separators=(",", ":"))
PY

set +e
HTTP="$(curl --max-time 180 -sS \
  -o "$ROOT/response.json" \
  -w '%{http_code}' \
  "http://127.0.0.1:${PORT}/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  --data-binary "@$ROOT/request.json")"
RC=$?
set -e

echo "mm_smoke_curl_rc=${RC}"
echo "mm_smoke_http=${HTTP}"

set +e
python3 - "$ROOT/response.json" "$HTTP" "$RC" <<'PY'
import json, re, sys
path, http, rc = sys.argv[1], sys.argv[2], int(sys.argv[3])
ok = rc == 0 and http == "200"
content = reasoning = finish = ""
try:
    d = json.load(open(path))
    choice = (d.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    content = str(msg.get("content") or "").strip()
    reasoning = str(msg.get("reasoning_content") or "").strip()
    finish = str(choice.get("finish_reason") or "")
    combined = " ".join(x for x in (reasoning, content) if x)
    print(f"mm_smoke_content={content!r}")
    print(f"mm_smoke_reasoning={reasoning!r}")
    print(f"mm_smoke_finish_reason={finish!r}")
    # Previous smoke accidentally used r"\\bred\\b", which searches for
    # literal backslashes instead of regex word boundaries and can never match
    # an ordinary answer like "red".
    semantic = bool(re.search(r"\bred\b", combined, re.I))
    print(f"mm_smoke_semantic_red={semantic}")
    ok = ok and semantic
except Exception as e:
    print(f"mm_smoke_parse_error={e!r}")
    ok = False
print(f"QWEN38_MM_SMOKE_PASS={ok}")
raise SystemExit(0 if ok else 1)
PY
VERIFY_RC=$?
set -e

if (( VERIFY_RC != 0 )); then
  echo '--- MM SMOKE FIRST ERROR CANDIDATES ---'
  docker logs "$CONTAINER" 2>&1 | \
    grep -Ei 'Traceback|Scheduler hit an exception|RuntimeError|AssertionError|AcceleratorError|CUDA error|out of memory|MTP-PP|multimodal|vision|image' | \
    tail -240 || true
  echo '--- MM SMOKE RESPONSE ---'
  cat "$ROOT/response.json" 2>/dev/null || true
  echo
  exit "$VERIFY_RC"
fi

echo "QWEN38_GITTENSOR_PP3_NVFP4_MM_SMOKE=PASS"
