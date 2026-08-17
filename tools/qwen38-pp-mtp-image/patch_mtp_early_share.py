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
'''

    s = s.replace(marker, replacement, 1)

EAGLE.write_text(s)

print("PATCHED native PP-MTP early sharing")
print(EAGLE)
