from pathlib import Path
import re

P = Path("/sgl-workspace/sglang/python/sglang/srt/managers/scheduler_pp_mixin.py")
s = P.read_text()

# The native PP serializer still treats every EagleDraftInput as a full draft
# state and serializes hidden_states. Our topk=1 PP bridge deliberately uses
# EagleDraftInput only as a tiny token-chain carrier, with hidden_states=None.
# Preserve the legacy path for ordinary draft inputs, but skip only the
# hidden-state field for the bridge carrier; __mtp_pp_verify_tokens is already
# emitted by patch_mtp_pp_spec_bridge.py.
pattern = re.compile(
    r'^(?P<indent>[ \t]*)tensor_dict\["draft_hidden_states"\] = '
    r'draft_input\.hidden_states\.contiguous\(\)\s*$',
    re.MULTILINE,
)
match = pattern.search(s)
if match is None:
    if "[MTP-PP-BRIDGE-SERIALIZER]" not in s:
        raise RuntimeError("legacy PP draft_hidden_states serializer line not found")
else:
    indent = match.group("indent")
    replacement = (
        f'{indent}# [MTP-PP-BRIDGE-SERIALIZER] Token-only bridge has no hidden state.\n'
        f'{indent}if draft_input.hidden_states is not None:\n'
        f'{indent}    tensor_dict["draft_hidden_states"] = '
        'draft_input.hidden_states.contiguous()'
    )
    s = pattern.sub(replacement, s, count=1)
    P.write_text(s)

# Semantic audit: the verify-token bridge must exist and every remaining legacy
# hidden-state serialization must be structurally guarded by the None check.
# Do not merely search for the assignment text: it is expected to remain inside
# the guarded legacy path.
s = P.read_text()
if "__mtp_pp_verify_tokens" not in s:
    raise RuntimeError("MTP PP verify-token serializer is missing")
if "[MTP-PP-BRIDGE-SERIALIZER]" not in s:
    raise RuntimeError("MTP PP bridge serializer guard was not installed")

lines = s.splitlines()
assignment = (
    'tensor_dict["draft_hidden_states"] = '
    "draft_input.hidden_states.contiguous()"
)
serializer_lines = [
    i for i, line in enumerate(lines)
    if line.strip() == assignment
]
if not serializer_lines:
    raise RuntimeError("PP draft_hidden_states legacy serializer disappeared unexpectedly")

for i in serializer_lines:
    if i == 0:
        raise RuntimeError("PP draft_hidden_states serializer has no guard")
    current_indent = len(lines[i]) - len(lines[i].lstrip())
    prev = lines[i - 1]
    prev_indent = len(prev) - len(prev.lstrip())
    if (
        prev.strip() != "if draft_input.hidden_states is not None:"
        or current_indent <= prev_indent
    ):
        excerpt = "\n".join(lines[max(0, i - 3): i + 2])
        raise RuntimeError(
            "PP draft_hidden_states serializer is not structurally guarded; "
            f"excerpt={excerpt!r}"
        )

print("PATCHED token-only MTP PP bridge serializer")
print("VERIFIED token-only bridge structurally guards legacy draft_hidden_states serialization")
