from pathlib import Path
import ast

P = Path(
    "/sgl-workspace/sglang/python/sglang/srt/mem_cache/kv_cache_configurator.py"
)
s = P.read_text()
MARKER = "[MTP-PP-LOCAL-MEM-BUDGET]"


def _attr_tail(node):
    parts = []
    cur = node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id)
    return ".".join(reversed(parts))


def _abs_offset(text, lineno, col):
    lines = text.splitlines(keepends=True)
    return sum(len(x) for x in lines[: lineno - 1]) + col


if MARKER not in s:
    tree = ast.parse(s, filename=str(P))
    candidates = []

    for cls in tree.body:
        if not isinstance(cls, ast.ClassDef) or cls.name != "KVCacheConfigurator":
            continue
        for fn in cls.body:
            if not isinstance(fn, ast.FunctionDef) or fn.name != "_profile_available_bytes":
                continue
            for node in ast.walk(fn):
                if not isinstance(node, ast.Assign):
                    continue
                if not any(
                    isinstance(t, ast.Name) and t.id == "available_gpu_memory"
                    for t in node.targets
                ):
                    continue
                call = node.value
                if not isinstance(call, ast.Call):
                    continue
                if not _attr_tail(call.func).endswith("get_available_gpu_memory"):
                    continue
                distributed_kw = next(
                    (kw for kw in call.keywords if kw.arg == "distributed"), None
                )
                if distributed_kw is None:
                    continue
                candidates.append((fn, node, distributed_kw))

    if len(candidates) != 1:
        desc = [
            (fn.name, int(assign.lineno), ast.get_source_segment(s, kw.value))
            for fn, assign, kw in candidates
        ]
        raise RuntimeError(
            "Expected exactly one _profile_available_bytes distributed-memory probe; "
            f"found {len(candidates)}: {desc}"
        )

    fn, assign, distributed_kw = candidates[0]
    old_expr = ast.get_source_segment(s, distributed_kw.value)
    if not old_expr:
        raise RuntimeError("Could not recover distributed= expression source")

    # TP=1 + PP>1: each PP stage must size from its own physical free memory.
    # The stage-specific token capacities are already synchronized safely by the
    # later ReduceOp.MIN in _apply_token_constraints().  Taking MIN of free
    # memory here first makes the MTP-heavy PP-last stage shrink every other
    # stage before their different KV cell sizes are even considered.
    new_expr = (
        "(False if (self.ps.pp_size > 1 and self.ps.tp_size == 1) "
        "else get_world_group().world_size > 1)"
    )
    start = _abs_offset(s, distributed_kw.value.lineno, distributed_kw.value.col_offset)
    end = _abs_offset(
        s,
        distributed_kw.value.end_lineno,
        distributed_kw.value.end_col_offset,
    )
    s = s[:start] + new_expr + s[end:]

    # The expression replacement does not change line count, so the original
    # assignment end line remains valid. Emit a runtime marker with the actual
    # local free-memory value chosen by the helper.
    lines = s.splitlines(keepends=True)
    assign_line = lines[assign.lineno - 1]
    indent = assign_line[: len(assign_line) - len(assign_line.lstrip())]
    insert_at = int(assign.end_lineno or assign.lineno)
    log_lines = [
        f'{indent}logger.info(\n',
        f'{indent}    "[MTP-PP-LOCAL-MEM-BUDGET] PP%d tp=%d pp=%d "\n',
        f'{indent}    "free_gpu_mem=%.2fGB distributed_probe=%s",\n',
        f'{indent}    int(self.ps.pp_rank),\n',
        f'{indent}    int(self.ps.tp_size),\n',
        f'{indent}    int(self.ps.pp_size),\n',
        f'{indent}    float(available_gpu_memory),\n',
        f'{indent}    bool(not (self.ps.pp_size > 1 and self.ps.tp_size == 1) and get_world_group().world_size > 1),\n',
        f'{indent})\n',
    ]
    lines[insert_at:insert_at] = log_lines
    s = "".join(lines)

    ast.parse(s, filename=str(P))
    P.write_text(s)

    print(
        "FOUND PP free-memory probe via AST: "
        f"function={fn.name} assignment_line={assign.lineno} old_distributed={old_expr!r}"
    )

s = P.read_text()
for required in (
    MARKER,
    "self.ps.pp_size > 1",
    "self.ps.tp_size == 1",
    "_apply_token_constraints",
):
    if required not in s:
        raise RuntimeError(f"PP local-memory budget patch missing: {required}")

# Preserve the final token-capacity collective. This patch only removes the
# earlier free-memory MIN for the experiment's TP1+PP topology.
if "ReduceOp.MIN" not in s or "token_capacity = tensor.item()" not in s:
    raise RuntimeError("Final PP token-capacity MIN synchronization is missing")

ast.parse(s, filename=str(P))
print("PATCHED TP1+PP local free-memory profiling")
print("VERIFIED final PP token-capacity MIN remains authoritative")
print(P)
