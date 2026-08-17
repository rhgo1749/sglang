#!/usr/bin/env python3
from pathlib import Path

PATH = Path(
    "/sgl-workspace/sglang/python/sglang/srt/speculative/eagle_worker_v2.py"
)

text = PATH.read_text()

old = '''                if self.ps.pp_size > 1:\n                    batch_output.next_draft_input = self._mtp_pp_precompute_bridge(\n                        batch,\n                        batch_output.next_draft_input,\n                        batch_output.new_seq_lens,\n                        reserve_after_prefill=True,\n                    )\n'''

new = '''                if (\n                    self.ps.pp_size > 1\n                    and getattr(batch, "contains_last_prefill_chunk", True)\n                ):\n                    # The PP bridge is the proposal for the *first decode/verify*\n                    # iteration.  Chunked prefill must not reserve decode KV or\n                    # publish a verify bridge for every intermediate chunk; doing\n                    # so lets PP stages transition request boundaries at different\n                    # times and can deadlock the pipeline.\n                    logger.info(\n                        "[MTP-PP-PREFILL-BRIDGE-FINAL] PP%d final prefill chunk",\n                        int(self.ps.pp_rank),\n                    )\n                    batch_output.next_draft_input = self._mtp_pp_precompute_bridge(\n                        batch,\n                        batch_output.next_draft_input,\n                        batch_output.new_seq_lens,\n                        reserve_after_prefill=True,\n                    )\n'''

if new not in text:
    count = text.count(old)
    if count != 1:
        raise SystemExit(
            "ERROR: expected exactly one unconditional PP prefill bridge block; "
            f"found {count}"
        )
    text = text.replace(old, new, 1)

# Build-time structural guards: the first-decode bridge must be gated by the
# scheduler's PP chunk marker, while preserving the actual reserve operation.
if 'contains_last_prefill_chunk' not in text:
    raise SystemExit("ERROR: final-prefill bridge gate was not installed")
if text.count('reserve_after_prefill=True') != 1:
    raise SystemExit("ERROR: unexpected reserve_after_prefill=True call count")
if '[MTP-PP-PREFILL-BRIDGE-FINAL]' not in text:
    raise SystemExit("ERROR: final-prefill bridge marker missing")

compile(text, str(PATH), "exec")
PATH.write_text(text)
print(
    "[MTP-PP-PREFILL-BRIDGE] gated first verify bridge to the final prefill chunk"
)
