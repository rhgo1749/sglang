from pathlib import Path

P = Path("/tmp/patch_mtp_pp_spec_bridge.py")
s = P.read_text()

needle = '''# The current base normally imports build_eagle_verify_input already. Fail fast\n# rather than silently calling an unavailable helper.\nif "build_eagle_verify_input" not in s[:class_at]:\n    raise RuntimeError("build_eagle_verify_input import missing from EAGLE worker")\n\nhelper_marker = "[MTP-PP-SPEC-BRIDGE]"\n'''
replacement = '''# Import edits above change every later byte offset. Re-resolve the actual class\n# and method before inserting helpers; never reuse pre-edit string offsets.\nclass_at = s.find("class EAGLEWorkerV2(")\nif class_at < 0:\n    raise RuntimeError("class EAGLEWorkerV2 disappeared after import patch")\nfn_at = s.find("    def forward_batch_generation(", class_at)\nif fn_at < 0:\n    raise RuntimeError("EAGLEWorkerV2.forward_batch_generation disappeared after import patch")\n\n# The current base normally imports build_eagle_verify_input already. Fail fast\n# rather than silently calling an unavailable helper.\nif "build_eagle_verify_input" not in s[:class_at]:\n    raise RuntimeError("build_eagle_verify_input import missing from EAGLE worker")\n\nhelper_marker = "[MTP-PP-SPEC-BRIDGE]"\n'''
if needle not in s:
    raise RuntimeError("bridge installer class-offset patch point not found")
s = s.replace(needle, replacement, 1)

# Append a semantic audit in addition to py_compile: the helper calls verify with
# pp_proxy_tensors, so its actual EAGLEWorkerV2.verify signature must accept it.
s += r'''

# Installer semantic audit added by fix_mtp_pp_spec_bridge_installer.py
_e = EAGLE.read_text()
_ca = _e.find("class EAGLEWorkerV2(")
_va = _e.find("    def verify(", _ca)
_ve = _e.find("\n    def ", _va + 8)
if _ve < 0:
    _ve = len(_e)
_vf = _e[_va:_ve]
if _va < 0 or "pp_proxy_tensors=None" not in _vf.split(":", 1)[0]:
    raise RuntimeError("EAGLEWorkerV2.verify did not gain pp_proxy_tensors")
if "pp_proxy_tensors=pp_proxy_tensors" not in _vf:
    raise RuntimeError("EAGLEWorkerV2.verify did not forward pp_proxy_tensors")
print("VERIFIED EAGLEWorkerV2.verify PP proxy signature")
'''

P.write_text(s)
print("FIXED bridge installer offsets and added semantic audit")
