from pathlib import Path

P = Path("/tmp/patch_mtp_pp_spec_bridge.py")
s = P.read_text()

# The qwen38-27b base's shared verify helper includes num_steps between topk
# and num_draft_tokens. Keep the generated bridge signature aligned with the
# actual helper instead of matching the older signature shape.
sig_old = '    topk: int,\\n    num_draft_tokens: int,\\n'
sig_new = '    topk: int,\\n    num_steps: int,\\n    num_draft_tokens: int,\\n'
if sig_old not in s:
    raise RuntimeError("bridge installer run_eagle_verify signature source point not found")
s = s.replace(sig_old, sig_new, 1)

needle = '''# The current base normally imports build_eagle_verify_input already. Fail fast\n# rather than silently calling an unavailable helper.\nif "build_eagle_verify_input" not in s[:class_at]:\n    raise RuntimeError("build_eagle_verify_input import missing from EAGLE worker")\n\nhelper_marker = "[MTP-PP-SPEC-BRIDGE]"\n'''
replacement = '''# Import edits above change every later byte offset. Re-resolve the actual class\n# and method before inserting helpers; never reuse pre-edit string offsets.\nclass_at = s.find("class EAGLEWorkerV2(")\nif class_at < 0:\n    raise RuntimeError("class EAGLEWorkerV2 disappeared after import patch")\nfn_at = s.find("    def forward_batch_generation(", class_at)\nif fn_at < 0:\n    raise RuntimeError("EAGLEWorkerV2.forward_batch_generation disappeared after import patch")\n\n# The current base normally imports build_eagle_verify_input already. Fail fast\n# rather than silently calling an unavailable helper.\nif "build_eagle_verify_input" not in s[:class_at]:\n    raise RuntimeError("build_eagle_verify_input import missing from EAGLE worker")\n\nhelper_marker = "[MTP-PP-SPEC-BRIDGE]"\n'''
if needle not in s:
    raise RuntimeError("bridge installer class-offset patch point not found")
s = s.replace(needle, replacement, 1)

# The prefill->first-decode bridge reserves the same KV window as the normal
# scheduler, but it runs one phase earlier.  With ordinary no-penalty sampling,
# SamplingBatchInfo exists while penalizer_orchestrator is still None.  The
# upstream helper dereferences `.is_required` unconditionally before doing the
# reserve.  Install a temporary no-op orchestrator only for that early reserve;
# preserve any real orchestrator and restore None immediately afterwards.
reserve_old = '''            decode_batch_idx = [int(r.decode_batch_idx) for r in batch.reqs]\n            eagle_prepare_for_decode(batch)\n            for req, old_idx in zip(batch.reqs, decode_batch_idx):\n                req.decode_batch_idx = old_idx\n'''
reserve_new = '''            decode_batch_idx = [int(r.decode_batch_idx) for r in batch.reqs]\n            _sampling_info = getattr(batch, "sampling_info", None)\n            if _sampling_info is None:\n                raise RuntimeError(\n                    "[MTP-PP-RESERVE-SAMPLING] sampling_info is missing before early decode reserve"\n                )\n            _penalizer = getattr(_sampling_info, "penalizer_orchestrator", None)\n            _installed_no_penalty = False\n            if _penalizer is None:\n                class _MTPPPNoPenaltyOrchestrator:\n                    is_required = False\n\n                _sampling_info.penalizer_orchestrator = _MTPPPNoPenaltyOrchestrator()\n                _installed_no_penalty = True\n                logger.info(\n                    "[MTP-PP-RESERVE-NOPENALTY] PP%d temporary no-op penalizer for early KV reserve",\n                    int(self.ps.pp_rank),\n                )\n            try:\n                eagle_prepare_for_decode(batch)\n            finally:\n                if _installed_no_penalty:\n                    _sampling_info.penalizer_orchestrator = None\n            for req, old_idx in zip(batch.reqs, decode_batch_idx):\n                req.decode_batch_idx = old_idx\n'''
if reserve_old not in s:
    at = s.find("eagle_prepare_for_decode(batch)")
    excerpt = s[max(0, at - 260): at + 340] if at >= 0 else "<reserve call not found>"
    raise RuntimeError(
        "bridge installer early-reserve patch-source point not found; excerpt="
        + repr(excerpt)
    )
s = s.replace(reserve_old, reserve_new, 1)

# patch_mtp_pp_spec_bridge.py stores the generated method body in a Python
# triple-quoted string. The source therefore contains literal backslash-n
# sequences, but the quotes themselves are ordinary source quotes. Accept the
# escaped-quote spelling too so this installer is insensitive to how the source
# was generated/serialized.
future_variants = [
    r'''        _next_gpu = pp_outputs["next_token_ids"].to(torch.int64)\n        self.future_map.stash(\n            batch.req_pool_indices, RelayPayload(bonus_tokens=_next_gpu)\n        )\n        batch.input_ids = None\n''',
    r'''        _next_gpu = pp_outputs[\"next_token_ids\"].to(torch.int64)\n        self.future_map.stash(\n            batch.req_pool_indices, RelayPayload(bonus_tokens=_next_gpu)\n        )\n        batch.input_ids = None\n''',
]
future_old = next((x for x in future_variants if x in s), None)
if future_old is None:
    # Give a useful diagnostic instead of another opaque exact-needle failure.
    at = s.find('self.future_map.stash(')
    excerpt = s[max(0, at - 180): at + 260] if at >= 0 else '<stash not found>'
    raise RuntimeError(
        "bridge installer FutureMap patch-source point not found; excerpt="
        + repr(excerpt)
    )
future_new = r'''        _next_gpu = pp_outputs["next_token_ids"].to(torch.int64)\n        if batch.spec_algorithm.is_none():\n            self.future_map.stash(\n                batch.req_pool_indices, RelayPayload(bonus_tokens=_next_gpu)\n            )\n        batch.input_ids = None\n'''
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
if "[MTP-PP-RESERVE-NOPENALTY]" not in _e:
    raise RuntimeError("EAGLEWorkerV2 early reserve lacks no-penalty shim")

_p = PP.read_text()
_future_guard = (
    '        _next_gpu = pp_outputs["next_token_ids"].to(torch.int64)\n'
    '        if batch.spec_algorithm.is_none():\n'
    '            self.future_map.stash(\n'
)
if _future_guard not in _p:
    raise RuntimeError("sync PP spec still writes unconditionally into FutureMap")

print("VERIFIED EAGLEWorkerV2.verify PP proxy signature")
print("VERIFIED EAGLEWorkerV2 early reserve no-penalty shim")
print("VERIFIED sync PP spec bypasses FutureMap token stash")
'''

P.write_text(s)
print("FIXED bridge installer signature, offsets, early-reserve sampling state, FutureMap routing, and semantic audits")
