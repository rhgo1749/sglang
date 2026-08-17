from pathlib import Path

P = Path("/sgl-workspace/sglang/python/sglang/srt/model_executor/pool_configurator.py")
s = P.read_text()

# The native PP3 MTP bridge owns the draft worker/KV pool only on PP-last.
# Upstream DefaultPoolConfigurator prices EAGLE draft KV into every target PP
# stage, which is correct for colocated speculative workers but becomes a
# phantom surcharge on PP0/PP1 in this bridge topology.  Keep the surcharge on
# PP-last only; target KV accounting on non-owner stages must reflect only the
# target layers physically allocated there.
old = '''        # EAGLE/STANDALONE: scale cell_size to account for draft model KV cache.\n        # Assumes draft and target share the same per-layer KV size (head_dim,\n        # num_kv_heads, dtype), which holds for EAGLE/MTP draft models that\n        # reuse the target architecture's attention config.\n        if (\n            kvc.spec_algorithm.is_eagle() or kvc.spec_algorithm.is_standalone()\n        ) and not kvc.is_draft_worker:\n            eagle_draft_num_layers = kvc.spec_aux_config.eagle_draft_num_layers\n            if (\n                eagle_draft_num_layers is not None\n                and int(eagle_draft_num_layers) > 0\n                and int(num_layers) > 0\n            ):\n                self._cell_size = int(\n                    self._cell_size\n                    * (1 + int(eagle_draft_num_layers) / int(num_layers))\n                )\n'''

new = '''        # EAGLE/STANDALONE: scale cell_size to account for draft model KV cache.\n        # Assumes draft and target share the same per-layer KV size (head_dim,\n        # num_kv_heads, dtype), which holds for EAGLE/MTP draft models that\n        # reuse the target architecture's attention config.\n        #\n        # [MTP-PP-POOL-PROFILE] The qwen38 native PP bridge used by this image\n        # owns the only draft worker/KV pool on PP-last.  Charging draft KV to\n        # PP0/PP1 creates a phantom capacity cap despite no draft pool existing\n        # there.  The image is bridge-specific, so only PP-last carries the\n        # speculative KV surcharge when PP is active.\n        _mtp_pp_draft_budget_owner = (\n            kvc.ps.pp_size <= 1 or kvc.ps.pp_rank == kvc.ps.pp_size - 1\n        )\n        if (\n            kvc.spec_algorithm.is_eagle() or kvc.spec_algorithm.is_standalone()\n        ) and not kvc.is_draft_worker:\n            eagle_draft_num_layers = kvc.spec_aux_config.eagle_draft_num_layers\n            if (\n                eagle_draft_num_layers is not None\n                and int(eagle_draft_num_layers) > 0\n                and int(num_layers) > 0\n            ):\n                if _mtp_pp_draft_budget_owner:\n                    self._cell_size = int(\n                        self._cell_size\n                        * (1 + int(eagle_draft_num_layers) / int(num_layers))\n                    )\n                    logger.info(\n                        "[MTP-PP-POOL-PROFILE] PP%d owns draft KV budget; "\n                        "layers=%d draft_layers=%d cell_size=%d",\n                        int(kvc.ps.pp_rank),\n                        int(num_layers),\n                        int(eagle_draft_num_layers),\n                        int(self._cell_size),\n                    )\n                else:\n                    logger.info(\n                        "[MTP-PP-POOL-PROFILE] PP%d skips phantom draft KV "\n                        "budget; layers=%d draft_layers=%d cell_size=%d",\n                        int(kvc.ps.pp_rank),\n                        int(num_layers),\n                        int(eagle_draft_num_layers),\n                        int(self._cell_size),\n                    )\n'''

if "[MTP-PP-POOL-PROFILE]" not in s:
    if old not in s:
        raise RuntimeError("DefaultPoolConfigurator EAGLE draft-KV pricing block not found")
    s = s.replace(old, new, 1)
    P.write_text(s)

s = P.read_text()
marker = "[MTP-PP-POOL-PROFILE]"
if marker not in s:
    raise RuntimeError("PP-aware MTP pool-profile marker missing")
window_start = s.find(marker)
window = s[window_start - 1200 : window_start + 2600]
for required in (
    "_mtp_pp_draft_budget_owner",
    "kvc.ps.pp_rank == kvc.ps.pp_size - 1",
    "skips phantom draft KV",
    "owns draft KV budget",
):
    if required not in window:
        raise RuntimeError(f"PP-aware pool-profile patch missing: {required}")

print("PATCHED PP-aware native MTP KV pool profiling")
print("VERIFIED only PP-last is charged the draft KV surcharge")
print(P)
