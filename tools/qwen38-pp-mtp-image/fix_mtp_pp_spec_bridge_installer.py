from pathlib import Path

P = Path("/tmp/patch_mtp_pp_spec_bridge.py")
s = P.read_text()

needle = '''# The current base normally imports build_eagle_verify_input already. Fail fast\n# rather than silently calling an unavailable helper.\nif "build_eagle_verify_input" not in s[:class_at]:\n    raise RuntimeError("build_eagle_verify_input import missing from EAGLE worker")\n\nhelper_marker = "[MTP-PP-SPEC-BRIDGE]"\n'''
replacement = '''# Import edits above change every later byte offset. Re-resolve the actual class\n# and method before inserting helpers; never reuse pre-edit string offsets.\nclass_at = s.find("class EAGLEWorkerV2(")\nif class_at < 0:\n    raise RuntimeError("class EAGLEWorkerV2 disappeared after import patch")\nfn_at = s.find("    def forward_batch_generation(", class_at)\nif fn_at < 0:\n    raise RuntimeError("EAGLEWorkerV2.forward_batch_generation disappeared after import patch")\n\n# The current base normally imports build_eagle_verify_input already. Fail fast\n# rather than silently calling an unavailable helper.\nif "build_eagle_verify_input" not in s[:class_at]:\n    raise RuntimeError("build_eagle_verify_input import missing from EAGLE worker")\n\nhelper_marker = "[MTP-PP-SPEC-BRIDGE]"\n'''
if needle not in s:
    raise RuntimeError("bridge installer class-offset patch point not found")
s = s.replace(needle, replacement, 1)

# patch_mtp_pp_spec_bridge.py contains the generated _pp_prep_batch_result body
# inside a Python triple-quoted string. Search the PATCH SOURCE representation
# (literal backslash-n sequences), not the generated target representation.
future_old = r'''        _next_gpu = pp_outputs["next_token_ids"].to(torch.int64)\n        self.future_map.stash(\n            batch.req_pool_indices, RelayPayload(bonus_tokens=_next_gpu)\n        )\n        batch.input_ids = None\n'''
future_new = r'''        _next_gpu = pp_outputs["next_token_ids"].to(torch.int64)\n        if batch.spec_algorithm.is_none():\n            self.future_map.stash(\n                batch.req_pool_indices, RelayPayload(bonus_tokens=_next_gpu)\n            )\n        batch.input_ids = None\n'''
if future_old not in s:
    raise RuntimeError("bridge installer FutureMap patch-source point not found")
s = s.replace(future_old, future_new, 1)

# Append semantic audits to the bridge patch itself. These run AFTER the bridge
# has modified the base-image files, so build failure points at a real semantic
# mismatch rather than allowing a half-installed image.
s += r'''

# Installer semantic audit added by fix_mtp_pp_spec_bridge_installer.py
_e = EAGLE.read_text()
_ca = _e.find("class EAGLEWorkerV2(")
if _ca < 0:
    raise RuntimeError("semantic audit: EAGLEWorkerV2 missing")
_va = _e.find("    def verify(", _ca)
if _va < 0:
    raise RuntimeError("semantic audit: EAGLEWorkerV2.verify missing")
_ve = _e.find("\n    def ", _va + len("    def verify("))
if _ve < 0:
    _ve = len(_e)
_vf = _e[_va:_ve]
_header_end = _vf.find("):")
if _header_end < 0:
    raise RuntimeError("semantic audit: EAGLEWorkerV2.verify header terminator missing")
_vheader = _vf[: _header_end + 2]
if "pp_proxy_tensors=None" not in _vheader:
    raise RuntimeError("EAGLEWorkerV2.verify did not gain pp_proxy_tensors")
if "pp_proxy_tensors=pp_proxy_tensors" not in _vf:
    raise RuntimeError("EAGLEWorkerV2.verify did not forward pp_proxy_tensors")

_p = PP.read_text()
_future_guard = (
    '        _next_gpu = pp_outputs["next_token_ids"].to(torch.int64)\n'
    '        if batch.spec_algorithm.is_none():\n'
    '            self.future_map.stash(\n'
)
if _future_guard not in _p:
    raise RuntimeError("sync PP spec still writes unconditionally into FutureMap")

print("VERIFIED EAGLEWorkerV2.verify PP proxy signature")
print("VERIFIED sync PP spec bypasses FutureMap token stash")
'''

P.write_text(s)
print("FIXED bridge installer syntax, source escaping, offsets, FutureMap routing, and semantic audits")
