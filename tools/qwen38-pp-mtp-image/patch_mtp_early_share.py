from pathlib import Path

EAGLE = Path(
    "/sgl-workspace/sglang/python/sglang/srt/speculative/eagle_worker_v2.py"
)

s = EAGLE.read_text()

marker = '''        # Alias for better readability
        self.draft_runner = self.draft_worker.model_runner
'''

if "[MTP-EARLY-SHARE]" not in s:
    if marker not in s:
        raise RuntimeError(
            "Could not find draft_runner assignment in eagle_worker_v2.py"
        )

    replacement = marker + '''
        # PP + native MTP experiment:
        # Share/drop temporary endpoint copies immediately after draft loading,
        # before target/draft memory-pool viability is calculated.
        _free_before_share, _total_before_share = torch.cuda.mem_get_info()

        self.init_token_map()
        self.init_lm_head()

        _free_after_share, _total_after_share = torch.cuda.mem_get_info()

        logger.info(
            "[MTP-EARLY-SHARE] PP%d free_before=%.2fGiB "
            "free_after=%.2fGiB released=%.2fGiB",
            int(get_pp_group().rank_in_group),
            _free_before_share / (1 << 30),
            _free_after_share / (1 << 30),
            (_free_after_share - _free_before_share) / (1 << 30),
        )

        # Diagnostic only: account physical CUDA storages after endpoint sharing.
        # named_parameters()/named_buffers() can contain aliases, so dedupe by
        # (device, storage data_ptr).  Also compare against target storages to
        # separate genuinely private MTP memory from target-shared aliases.
        _target_storage_keys = set()
        for _target_tensor in list(
            self.target_worker.model_runner.model.parameters()
        ) + list(self.target_worker.model_runner.model.buffers()):
            try:
                if _target_tensor.device.type != "cuda":
                    continue
                _target_storage = _target_tensor.untyped_storage()
                _target_ptr = int(_target_storage.data_ptr())
                if _target_ptr:
                    _target_storage_keys.add(
                        (int(_target_tensor.device.index or 0), _target_ptr)
                    )
            except Exception:
                pass

        _seen_storage_keys = set()
        _category_bytes = {}
        _private_rows = []
        _unique_total = 0
        _unique_shared = 0
        _unique_private = 0

        _draft_named_tensors = list(self.draft_runner.model.named_parameters())
        _draft_named_tensors += list(self.draft_runner.model.named_buffers())
        for _name, _tensor in _draft_named_tensors:
            try:
                if _tensor.device.type != "cuda":
                    continue
                _storage = _tensor.untyped_storage()
                _ptr = int(_storage.data_ptr())
                if not _ptr:
                    continue
                _key = (int(_tensor.device.index or 0), _ptr)
                if _key in _seen_storage_keys:
                    continue
                _seen_storage_keys.add(_key)
                _bytes = int(_storage.nbytes())
            except Exception:
                continue

            if "embed_tokens" in _name:
                _category = "embed"
            elif _name.startswith("lm_head") or ".lm_head" in _name:
                _category = "lm_head"
            elif _name.startswith("fc.") or "pre_fc_norm" in _name:
                _category = "mtp_adapter"
            else:
                _category = "mtp_body"

            _shared = _key in _target_storage_keys
            _unique_total += _bytes
            if _shared:
                _unique_shared += _bytes
            else:
                _unique_private += _bytes

            _cat = _category_bytes.setdefault(
                _category, {"total": 0, "shared": 0, "private": 0}
            )
            _cat["total"] += _bytes
            _cat["shared" if _shared else "private"] += _bytes

            if not _shared:
                _private_rows.append(
                    (
                        _bytes,
                        _name,
                        str(_tensor.dtype),
                        tuple(_tensor.shape),
                    )
                )

        logger.info(
            "[MTP-MEM-AUDIT] PP%d unique_total=%.3fGiB shared_with_target=%.3fGiB "
            "private=%.3fGiB cuda_allocated=%.3fGiB cuda_reserved=%.3fGiB",
            int(get_pp_group().rank_in_group),
            _unique_total / (1 << 30),
            _unique_shared / (1 << 30),
            _unique_private / (1 << 30),
            torch.cuda.memory_allocated() / (1 << 30),
            torch.cuda.memory_reserved() / (1 << 30),
        )
        for _category in ("embed", "lm_head", "mtp_adapter", "mtp_body"):
            _cat = _category_bytes.get(
                _category, {"total": 0, "shared": 0, "private": 0}
            )
            logger.info(
                "[MTP-MEM-AUDIT] category=%s total=%.3fGiB shared=%.3fGiB private=%.3fGiB",
                _category,
                _cat["total"] / (1 << 30),
                _cat["shared"] / (1 << 30),
                _cat["private"] / (1 << 30),
            )

        _private_rows.sort(key=lambda _row: _row[0], reverse=True)
        for _bytes, _name, _dtype, _shape in _private_rows[:24]:
            logger.info(
                "[MTP-MEM-TOP] %.1fMiB dtype=%s shape=%s name=%s",
                _bytes / (1 << 20),
                _dtype,
                _shape,
                _name,
            )
'''

    s = s.replace(marker, replacement, 1)

# Semantic audit: the diagnostic must be downstream of sharing, so it cannot
# perturb which endpoints are retained/released.
if "[MTP-MEM-AUDIT]" not in s or "[MTP-MEM-TOP]" not in s:
    raise RuntimeError("MTP memory audit injection missing")
if s.find("self.init_lm_head()") > s.find("[MTP-MEM-AUDIT]"):
    raise RuntimeError("MTP memory audit must run after init_lm_head sharing")

EAGLE.write_text(s)

print("PATCHED native PP-MTP early sharing + physical storage audit")
print("VERIFIED MTP memory audit runs after endpoint sharing")
print(EAGLE)
