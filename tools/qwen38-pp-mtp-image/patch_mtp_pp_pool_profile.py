from pathlib import Path

P = Path(
    "/sgl-workspace/sglang/python/sglang/srt/mem_cache/kv_cache_configurator.py"
)
s = P.read_text()

# Diagnostic only. Each PP rank profiles its own token capacity, then SGLang
# all-reduces MIN across PP so every stage allocates the same token count. The
# public max_total_num_tokens log is therefore the global minimum and hides the
# local headroom of PP0/PP1. Log both sides without replacing any arithmetic,
# branch, or collective statement.
LOCAL = "[MTP-PP-CAPACITY-LOCAL]"
GLOBAL = "[MTP-PP-CAPACITY-GLOBAL]"

if LOCAL not in s:
    # Do not depend on the exact formatting of the whole block in the pinned
    # base image. Locate the unique ReduceOp.MIN that operates on token_capacity
    # inside the configured-PP synchronization path.
    candidates = []
    cursor = 0
    min_needle = "op=torch.distributed.ReduceOp.MIN"
    while True:
        pos = s.find(min_needle, cursor)
        if pos < 0:
            break
        before = s[max(0, pos - 700) : pos]
        after = s[pos : pos + 700]
        if (
            "configured_pp_size() > 1" in before
            and "token_capacity" in before
            and "token_capacity = tensor.item()" in after
        ):
            candidates.append(pos)
        cursor = pos + len(min_needle)

    if len(candidates) != 1:
        excerpts = [
            s[max(0, p - 260) : p + 420] for p in candidates[:4]
        ]
        raise RuntimeError(
            "Expected exactly one PP token-capacity MIN reduction; "
            f"found {len(candidates)} candidates: {excerpts}"
        )

    reduce_pos = candidates[0]

    tensor_needle = "tensor = torch.tensor(token_capacity, dtype=torch.int64)"
    tensor_pos = s.rfind(
        tensor_needle,
        max(0, reduce_pos - 700),
        reduce_pos,
    )
    if tensor_pos < 0:
        raise RuntimeError("token_capacity tensor creation not found before PP MIN")
    tensor_line_start = s.rfind("\n", 0, tensor_pos) + 1
    tensor_indent = s[tensor_line_start:tensor_pos]
    if not tensor_indent.isspace():
        raise RuntimeError(
            f"Unexpected tensor-line indentation before PP MIN: {tensor_indent!r}"
        )

    local_log = (
        f'{tensor_indent}logger.info(\n'
        f'{tensor_indent}    "[MTP-PP-CAPACITY-LOCAL] PP%d local_tokens=%d "\n'
        f'{tensor_indent}    "layers=%d:%d available_gpu_mem=%.2fGB",\n'
        f'{tensor_indent}    int(self.ps.pp_rank),\n'
        f'{tensor_indent}    int(token_capacity),\n'
        f'{tensor_indent}    int(self.layer_info.start_layer),\n'
        f'{tensor_indent}    int(self.layer_info.end_layer),\n'
        f'{tensor_indent}    get_available_gpu_memory(self.device, self.gpu_id),\n'
        f'{tensor_indent})\n'
    )
    s = s[:tensor_line_start] + local_log + s[tensor_line_start:]

    # Re-find after insertion so offsets cannot go stale.
    reduce_pos = s.find(min_needle, tensor_line_start)
    assign_needle = "token_capacity = tensor.item()"
    assign_pos = s.find(assign_needle, reduce_pos, reduce_pos + 900)
    if assign_pos < 0:
        raise RuntimeError("token_capacity MIN assignment not found after PP reduction")
    assign_line_start = s.rfind("\n", 0, assign_pos) + 1
    assign_indent = s[assign_line_start:assign_pos]
    assign_line_end = s.find("\n", assign_pos)
    if assign_line_end < 0:
        raise RuntimeError("token_capacity MIN assignment line has no newline")
    if assign_indent != tensor_indent:
        raise RuntimeError(
            "PP MIN tensor/assignment indentation mismatch: "
            f"{tensor_indent!r} vs {assign_indent!r}"
        )

    global_log = (
        f'{assign_indent}logger.info(\n'
        f'{assign_indent}    "[MTP-PP-CAPACITY-GLOBAL] PP%d global_min_tokens=%d",\n'
        f'{assign_indent}    int(self.ps.pp_rank),\n'
        f'{assign_indent}    int(token_capacity),\n'
        f'{assign_indent})\n'
    )
    insert_at = assign_line_end + 1
    s = s[:insert_at] + global_log + s[insert_at:]
    P.write_text(s)

s = P.read_text()
for required in (
    LOCAL,
    GLOBAL,
    "torch.distributed.ReduceOp.MIN",
    "self.layer_info.start_layer",
    "self.layer_info.end_layer",
):
    if required not in s:
        raise RuntimeError(f"PP capacity diagnostic missing: {required}")

if s.count(LOCAL) != 1 or s.count(GLOBAL) != 1:
    raise RuntimeError(
        "PP capacity diagnostics must be injected exactly once: "
        f"local={s.count(LOCAL)} global={s.count(GLOBAL)}"
    )

# This patch must not resurrect the rejected phantom-draft-KV pricing change.
if "_mtp_pp_draft_budget_owner" in s or "skips phantom draft KV" in s:
    raise RuntimeError("obsolete phantom draft-KV pricing patch is still present")

print("PATCHED PP local/global KV capacity diagnostics")
print("VERIFIED capacity arithmetic unchanged; logging wraps PP MIN reduction only")
print(P)
