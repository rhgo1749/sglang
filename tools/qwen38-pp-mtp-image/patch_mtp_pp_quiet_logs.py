from pathlib import Path
import re

ROOT = Path("/sgl-workspace/sglang/python/sglang/srt")
FILES = (
    ROOT / "managers/scheduler.py",
    ROOT / "managers/scheduler_pp_mixin.py",
    ROOT / "speculative/eagle_worker_v2.py",
    ROOT / "models/qwen3_5_mtp.py",
)

# Keep boot/capacity/memory-audit markers at INFO.  These markers fire on every
# 512/1024-token prefill chunk or every speculative decode step and were useful
# while bringing PP3+MTP up, but they materially perturb a production benchmark
# and generate very large logs at 256K context.  Errors remain RuntimeErrors and
# therefore stay visible to the fail-fast watcher.
HOT_MARKERS = (
    "MTP-PP-PHASE",
    "MTP-PP-IDENTITY",
    "MTP-PP-REQ-SOURCE",
    "MTP-PP-TRANSPORT-IDS",
    "MTP-PP-INPUT-RANGE",
    "MTP-PP-RESERVE-NOPENALTY",
    "MTP-PP-SPEC-BRIDGE",
    "MTP-PP-BRIDGE-RECV",
    "MTP-PP-BRIDGE-BUILD",
    "MTP-PP-VERIFY-STAGE",
    "MTP-PP-VERIFY-LAST",
    "MTP-PP-MAMBA-COMMIT",
    "MTP-PP-PREFILL-OWNER-V2",
    "MTP-PP-RUN-BRIDGE",
)

converted = 0
seen = set()
for path in FILES:
    text = path.read_text()
    before = text
    for marker in HOT_MARKERS:
        # All bridge diagnostics use a literal first argument beginning with the
        # marker.  Restrict the rewrite to that shape so unrelated INFO logs are
        # untouched.
        pat = re.compile(
            r"logger\.info\((\s*)([fFrRuUbB]*[\"']\[" + re.escape(marker) + r"\])"
        )
        text, n = pat.subn(r"logger.debug(\1\2", text)
        if n:
            converted += n
            seen.add(marker)
    if text != before:
        path.write_text(text)

required = {
    "MTP-PP-PHASE",
    "MTP-PP-REQ-SOURCE",
    "MTP-PP-SPEC-BRIDGE",
    "MTP-PP-BRIDGE-RECV",
}
missing = sorted(required - seen)
if missing:
    raise RuntimeError(f"hot-path log markers not found for quieting: {missing}")

print(
    f"[MTP-PP-QUIET] downgraded {converted} hot-path INFO diagnostics to DEBUG; "
    "boot/capacity/memory audit logs remain INFO"
)
