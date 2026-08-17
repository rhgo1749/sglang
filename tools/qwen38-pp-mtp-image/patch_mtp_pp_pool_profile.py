from pathlib import Path

P = Path(
    "/sgl-workspace/sglang/python/sglang/srt/mem_cache/kv_cache_configurator.py"
)
s = P.read_text()

# Diagnostic only.  Each PP rank profiles its own token capacity, then SGLang
# all-reduces MIN across PP so every stage allocates the same token count.  The
# public max_total_num_tokens log is therefore the global minimum and hides the
# local headroom of PP0/PP1.  Log both sides of that reduction so repartition and
# PP-last MTP memory work can be sized from the real per-stage capacities.
old = '''        # Sync across PP ranks (each may have different layer counts)\n        if configured_pp_size() > 1:\n            tensor = torch.tensor(token_capacity, dtype=torch.int64)\n            torch.distributed.all_reduce(\n                tensor,\n                op=torch.distributed.ReduceOp.MIN,\n                group=get_world_group().cpu_group,\n            )\n            token_capacity = tensor.item()\n\n        return token_capacity\n'''

new = '''        # Sync across PP ranks (each may have different layer counts)\n        if configured_pp_size() > 1:\n            logger.info(\n                "[MTP-PP-CAPACITY-LOCAL] PP%d local_tokens=%d layers=%d:%d "\n                "available_gpu_mem=%.2fGB",\n                int(self.ps.pp_rank),\n                int(token_capacity),\n                int(self.layer_info.start_layer),\n                int(self.layer_info.end_layer),\n                get_available_gpu_memory(self.device, self.gpu_id),\n            )\n            tensor = torch.tensor(token_capacity, dtype=torch.int64)\n            torch.distributed.all_reduce(\n                tensor,\n                op=torch.distributed.ReduceOp.MIN,\n                group=get_world_group().cpu_group,\n            )\n            token_capacity = tensor.item()\n            logger.info(\n                "[MTP-PP-CAPACITY-GLOBAL] PP%d global_min_tokens=%d",\n                int(self.ps.pp_rank),\n                int(token_capacity),\n            )\n\n        return token_capacity\n'''

if "[MTP-PP-CAPACITY-LOCAL]" not in s:
    if old not in s:
        raise RuntimeError("PP capacity MIN-reduction block not found")
    s = s.replace(old, new, 1)
    P.write_text(s)

s = P.read_text()
for required in (
    "[MTP-PP-CAPACITY-LOCAL]",
    "[MTP-PP-CAPACITY-GLOBAL]",
    "torch.distributed.ReduceOp.MIN",
    "self.layer_info.start_layer",
    "self.layer_info.end_layer",
):
    if required not in s:
        raise RuntimeError(f"PP capacity diagnostic missing: {required}")

# This patch must not change the capacity arithmetic itself.
if "_mtp_pp_draft_budget_owner" in s or "skips phantom draft KV" in s:
    raise RuntimeError("obsolete phantom draft-KV pricing patch is still present")

print("PATCHED PP local/global KV capacity diagnostics")
print("VERIFIED capacity arithmetic unchanged; logging wraps PP MIN reduction only")
print(P)
