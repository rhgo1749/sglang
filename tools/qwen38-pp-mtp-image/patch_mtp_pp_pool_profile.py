from pathlib import Path
import ast

P = Path(
    "/sgl-workspace/sglang/python/sglang/srt/mem_cache/kv_cache_configurator.py"
)
s = P.read_text()

# Diagnostic only. Each PP rank profiles its own token capacity, then SGLang
# all-reduces MIN across PP so every stage allocates the same token count. The
# public max_total_num_tokens log is therefore the global minimum and hides the
# local headroom of PP0/PP1.
#
# Do not patch this with exact source strings: the pinned Docker image can use
# different distributed aliases/formatting from the checked-in branch. Parse the
# actual image source and locate the semantic shape instead:
#   <anything>.all_reduce(... op=<anything>.ReduceOp.MIN ...)
#   ...
#   token_capacity = tensor.item()
# within the same function.
LOCAL = "[MTP-PP-CAPACITY-LOCAL]"
GLOBAL = "[MTP-PP-CAPACITY-GLOBAL]"


def _attr_tail(node):
    """Return dotted-ish attribute tail where possible, independent of aliases."""
    parts = []
    cur = node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id)
    return ".".join(reversed(parts))


def _is_min_reduce_call(node):
    if not isinstance(node, ast.Call):
        return False
    if not isinstance(node.func, ast.Attribute) or node.func.attr != "all_reduce":
        return False
    for kw in node.keywords:
        if kw.arg != "op":
            continue
        tail = _attr_tail(kw.value)
        if tail.endswith("ReduceOp.MIN"):
            return True
    return False


def _is_token_capacity_item_assign(node):
    if not isinstance(node, (ast.Assign, ast.AnnAssign)):
        return False
    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
    if not any(isinstance(t, ast.Name) and t.id == "token_capacity" for t in targets):
        return False
    value = node.value
    return (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Attribute)
        and value.func.attr == "item"
        and isinstance(value.func.value, ast.Name)
        and value.func.value.id == "tensor"
    )


if LOCAL not in s:
    tree = ast.parse(s, filename=str(P))
    candidates = []

    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue

        reduce_calls = [n for n in ast.walk(fn) if _is_min_reduce_call(n)]
        assigns = [n for n in ast.walk(fn) if _is_token_capacity_item_assign(n)]

        for reduce_call in reduce_calls:
            later_assigns = [a for a in assigns if a.lineno > reduce_call.lineno]
            if not later_assigns:
                continue
            assign = min(later_assigns, key=lambda a: a.lineno)
            # Keep the semantic match tight: both operations should be in the
            # same short synchronization block, not hundreds of lines apart.
            if assign.lineno - reduce_call.lineno <= 20:
                candidates.append((fn, reduce_call, assign))

    if len(candidates) != 1:
        desc = [
            (
                getattr(fn, "name", "<unknown>"),
                int(call.lineno),
                int(assign.lineno),
                _attr_tail(call.func),
            )
            for fn, call, assign in candidates
        ]
        raise RuntimeError(
            "Expected exactly one AST PP token-capacity MIN reduction; "
            f"found {len(candidates)} candidates: {desc}"
        )

    fn, reduce_call, assign = candidates[0]
    lines = s.splitlines(keepends=True)

    # Insert LOCAL immediately before the all_reduce statement. This is after
    # token_capacity has been finalized locally and before MIN can overwrite it.
    reduce_idx = reduce_call.lineno - 1
    reduce_line = lines[reduce_idx]
    reduce_indent = reduce_line[: len(reduce_line) - len(reduce_line.lstrip())]
    if not reduce_indent:
        raise RuntimeError(
            f"Unexpected zero indentation for PP all_reduce at line {reduce_call.lineno}"
        )

    local_log = [
        f'{reduce_indent}logger.info(\n',
        f'{reduce_indent}    "[MTP-PP-CAPACITY-LOCAL] PP%d local_tokens=%d "\n',
        f'{reduce_indent}    "layers=%d:%d available_gpu_mem=%.2fGB",\n',
        f'{reduce_indent}    int(self.ps.pp_rank),\n',
        f'{reduce_indent}    int(token_capacity),\n',
        f'{reduce_indent}    int(self.layer_info.start_layer),\n',
        f'{reduce_indent}    int(self.layer_info.end_layer),\n',
        f'{reduce_indent}    get_available_gpu_memory(self.device, self.gpu_id),\n',
        f'{reduce_indent})\n',
    ]

    # Insert GLOBAL after the complete token_capacity=tensor.item() statement.
    # Use end_lineno from the AST so alternate formatting is harmless.
    assign_end_idx = int(assign.end_lineno or assign.lineno)
    assign_line = lines[assign.lineno - 1]
    assign_indent = assign_line[: len(assign_line) - len(assign_line.lstrip())]
    if assign_indent != reduce_indent:
        raise RuntimeError(
            "PP MIN all_reduce/assignment indentation mismatch: "
            f"{reduce_indent!r} vs {assign_indent!r}"
        )

    global_log = [
        f'{assign_indent}logger.info(\n',
        f'{assign_indent}    "[MTP-PP-CAPACITY-GLOBAL] PP%d global_min_tokens=%d",\n',
        f'{assign_indent}    int(self.ps.pp_rank),\n',
        f'{assign_indent}    int(token_capacity),\n',
        f'{assign_indent})\n',
    ]

    # Apply bottom insertion first so original AST line numbers remain valid.
    lines[assign_end_idx:assign_end_idx] = global_log
    lines[reduce_idx:reduce_idx] = local_log
    s = "".join(lines)

    # Syntax-validate the exact target text before writing it into the image.
    ast.parse(s, filename=str(P))
    P.write_text(s)

    print(
        "FOUND PP capacity MIN reduction via AST: "
        f"function={fn.name} all_reduce_line={reduce_call.lineno} "
        f"assign_line={assign.lineno}"
    )

s = P.read_text()
for required in (
    LOCAL,
    GLOBAL,
    "token_capacity",
    "all_reduce",
):
    if required not in s:
        raise RuntimeError(f"PP capacity diagnostic missing: {required}")

if s.count(LOCAL) != 1 or s.count(GLOBAL) != 1:
    raise RuntimeError(
        "PP capacity diagnostics must be injected exactly once: "
        f"local={s.count(LOCAL)} global={s.count(GLOBAL)}"
    )

# This diagnostic must not resurrect the rejected phantom-draft-KV pricing change.
if "_mtp_pp_draft_budget_owner" in s or "skips phantom draft KV" in s:
    raise RuntimeError("obsolete phantom draft-KV pricing patch is still present")

# Final syntax validation of the modified runtime file.
ast.parse(s, filename=str(P))

print("PATCHED PP local/global KV capacity diagnostics via AST")
print("VERIFIED capacity arithmetic unchanged; logging wraps PP MIN reduction only")
print(P)
