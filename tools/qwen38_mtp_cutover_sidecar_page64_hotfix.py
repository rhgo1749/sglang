#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pathlib
import subprocess

HOME = pathlib.Path.home()
REPO = HOME / "projects/sglang-fork"
HOST = HOME / "projects/sglang-patches/eagle_worker_v2.sidecar-pool-probe.py"
FORK = REPO / "python/sglang/srt/speculative/eagle_worker_v2.py"

PAGE64_MARKER = '"[MTP-CUTOVER-PAGED] CUDA%d page_size=%d high-water allocator enabled"'

HELPER = r'''    def _mtp_authoritative_reserve_span(
        self, side_reqs, start_lens, end_lens, *, label
    ):
        """Return KV slots for [start,end), growing each request's paged high-water.

        For page_size>1, speculative slots may share a page with committed KV.  Such
        tails must not be individually freed: PagedTokenToKVPoolAllocator.free()
        returns whole pages.  Keep a per-request allocation high-water instead and
        reuse/overwrite speculative tail slots until the request is released.
        """
        if not (len(side_reqs) == len(start_lens) == len(end_lens)):
            raise RuntimeError(
                f"CUDA2 MTP {label} span batch mismatch: "
                f"reqs={len(side_reqs)} start={len(start_lens)} end={len(end_lens)}"
            )

        alloc_lens = [
            int(self._mtp_side_alloc_lens.get(req.rid, 0)) for req in side_reqs
        ]
        for req, start, end, alloc_len in zip(
            side_reqs, start_lens, end_lens, alloc_lens
        ):
            if start < 0 or end < start:
                raise RuntimeError(
                    f"CUDA2 MTP {label} invalid span rid={req.rid}: "
                    f"start={start} end={end}"
                )
            if start > alloc_len:
                raise RuntimeError(
                    f"CUDA2 MTP {label} allocation hole rid={req.rid}: "
                    f"start={start} alloc={alloc_len}"
                )

        grow_to = [max(a, int(e)) for a, e in zip(alloc_lens, end_lens)]
        grow_lens = [b - a for a, b in zip(alloc_lens, grow_to)]
        total_grow = int(sum(grow_lens))
        allocator = self.token_to_kv_pool_allocator

        if total_grow > 0:
            if int(getattr(allocator, "page_size", 1)) == 1:
                new_slots = allocator.alloc(total_grow)
            else:
                prefix_cpu = torch.tensor(alloc_lens, dtype=torch.int64)
                seq_cpu = torch.tensor(grow_to, dtype=torch.int64)
                prefix_dev = prefix_cpu.to(self.device)
                seq_dev = seq_cpu.to(self.device)
                last = []
                for req, alloc_len in zip(side_reqs, alloc_lens):
                    if alloc_len > 0:
                        last.append(
                            self.req_to_token_pool.req_to_token[
                                req.req_pool_idx, alloc_len - 1 : alloc_len
                            ]
                        )
                    else:
                        last.append(
                            torch.full(
                                (1,), -1, dtype=torch.int64, device=self.device
                            )
                        )
                new_slots = allocator.alloc_extend(
                    prefix_dev,
                    prefix_cpu,
                    seq_dev,
                    seq_cpu,
                    torch.cat(last),
                    total_grow,
                )

            if new_slots is None:
                raise RuntimeError(
                    f"CUDA2 MTP {label} KV allocation failed: need={total_grow} "
                    f"page_size={getattr(allocator, 'page_size', None)}"
                )

            off = 0
            for req, old_alloc, new_alloc, grow in zip(
                side_reqs, alloc_lens, grow_to, grow_lens
            ):
                if grow:
                    self.req_to_token_pool.write(
                        (req.req_pool_idx, slice(old_alloc, new_alloc)),
                        new_slots[off : off + grow],
                    )
                    off += grow
                self._mtp_side_alloc_lens[req.rid] = new_alloc

        pieces = []
        for req, start, end in zip(side_reqs, start_lens, end_lens):
            if end > start:
                pieces.append(
                    self.req_to_token_pool.req_to_token[
                        req.req_pool_idx, start:end
                    ].clone()
                )
        if not pieces:
            return torch.empty((0,), dtype=torch.int64, device=self.device)
        out = torch.cat(pieces)
        expected = int(sum(e - s for s, e in zip(start_lens, end_lens)))
        if int(out.numel()) != expected:
            raise RuntimeError(
                f"CUDA2 MTP {label} span size mismatch: {out.numel()} != {expected}"
            )
        return out

'''

PREFILL_OLD = r'''            total_extend = int(sum(extend_lens))
            slots = self.token_to_kv_pool_allocator.alloc(total_extend)
            if slots is None:
                raise RuntimeError(
                    f"CUDA2 MTP prefill KV allocation failed: need={total_extend}"
                )
            off = 0
            for side_req, old_len, seq_len, extend_len in zip(
                side_reqs, old_lens, seq_lens, extend_lens
            ):
                if extend_len:
                    self.req_to_token_pool.write(
                        (side_req.req_pool_idx, slice(old_len, seq_len)),
                        slots[off : off + extend_len],
                    )
                off += extend_len
'''
PREFILL_NEW = r'''            total_extend = int(sum(extend_lens))
            slots = self._mtp_authoritative_reserve_span(
                side_reqs, old_lens, seq_lens, label="prefill"
            )
'''

DRAFT_OLD = r'''            future = self.token_to_kv_pool_allocator.alloc(bs * self.speculative_num_steps)
            if future is None:
                raise RuntimeError(
                    "CUDA2 MTP speculative KV reservation failed: "
                    f"need={bs * self.speculative_num_steps}"
                )
            off = 0
            for side_req, seq_len in zip(side_reqs, seq_lens):
                req_future = future[off : off + self.speculative_num_steps]
                self.req_to_token_pool.write(
                    (
                        side_req.req_pool_idx,
                        slice(seq_len, seq_len + self.speculative_num_steps),
                    ),
                    req_future,
                )
                off += self.speculative_num_steps
            self._mtp_side_last_draft_slots = future
'''
DRAFT_NEW = r'''            future = self._mtp_authoritative_reserve_span(
                side_reqs,
                seq_lens,
                [x + self.speculative_num_steps for x in seq_lens],
                label="draft",
            )
            self._mtp_side_last_draft_slots = future
'''

EXTEND_OLD = r'''            # Draft-decode's temporary KV is no longer needed after target verify.
            if self._mtp_side_last_draft_slots is not None:
                self.token_to_kv_pool_allocator.free(self._mtp_side_last_draft_slots)
                self._mtp_side_last_draft_slots = None

            width = self.speculative_num_draft_tokens
            extend_slots = self.token_to_kv_pool_allocator.alloc(bs * width)
            if extend_slots is None:
                raise RuntimeError(
                    f"CUDA2 MTP draft-extend KV allocation failed: need={bs * width}"
                )
            slot_rows = extend_slots.view(bs, width)
            for side_req, old_len, row_slots in zip(side_reqs, old_lens, slot_rows):
                self.req_to_token_pool.write(
                    (side_req.req_pool_idx, slice(old_len, old_len + width)),
                    row_slots,
                )
'''
EXTEND_NEW = r'''            # Paged speculative tails can share a page with committed KV, so never
            # free the temporary 3-token draft span independently.  Reuse it via
            # the per-request high-water and grow only when the 4-token extend crosses it.
            self._mtp_side_last_draft_slots = None

            width = self.speculative_num_draft_tokens
            extend_slots = self._mtp_authoritative_reserve_span(
                side_reqs,
                old_lens,
                [x + width for x in old_lens],
                label="draft-extend",
            )
            slot_rows = extend_slots.view(bs, width)
'''

TAIL_OLD = r'''            free_parts = []
            new_lens = []
            for target_req, side_req, old_len, accepted, row_slots in zip(
                target_batch.reqs, side_reqs, old_lens, accept_cpu, slot_rows
            ):
                if accepted < width:
                    free_parts.append(row_slots[accepted:])
                new_len = old_len + accepted
                new_lens.append(new_len)
                side_req.kv_committed_len = new_len
                self._mtp_side_seq_lens[target_req.rid] = new_len
            if free_parts:
                tail = torch.cat([x for x in free_parts if x.numel() > 0])
                if tail.numel() > 0:
                    self.token_to_kv_pool_allocator.free(tail)
'''
TAIL_NEW = r'''            new_lens = []
            for target_req, side_req, old_len, accepted in zip(
                target_batch.reqs, side_reqs, old_lens, accept_cpu
            ):
                new_len = old_len + accepted
                new_lens.append(new_len)
                side_req.kv_committed_len = new_len
                self._mtp_side_seq_lens[target_req.rid] = new_len
'''


def once(text: str, old: str, new: str, label: str, path: pathlib.Path) -> tuple[str, bool]:
    if old in text:
        return text.replace(old, new, 1), True
    if new in text:
        return text, False
    raise RuntimeError(f"{label} patch point not found: {path}")


def patch(path: pathlib.Path) -> bool:
    text = path.read_text()
    if PAGE64_MARKER in text:
        print(f"CUDA2 paged high-water patch already installed: {path}")
        return False

    changed = False

    text, c = once(
        text,
        "get_schedule().override(page_size=1),",
        "get_schedule().override(page_size=64),",
        "schedule page size",
        path,
    )
    changed |= c
    text, c = once(
        text,
        'object.__setattr__(_side_server_args, "page_size", 1)',
        'object.__setattr__(_side_server_args, "page_size", 64)',
        "private sidecar args page size",
        path,
    )
    changed |= c
    text, c = once(
        text,
        'if int(getattr(mr, "page_size", -1)) != 1:',
        'if int(getattr(mr, "page_size", -1)) != 64:',
        "sidecar page assertion",
        path,
    )
    changed |= c
    text = text.replace(
        '"CUDA2 MTP sidecar must use page_size=1; got "',
        '"CUDA2 MTP sidecar must use page_size=64; got "',
        1,
    )

    if "self._mtp_side_alloc_lens = {}" not in text:
        needle = "        self._mtp_side_seq_lens = {}\n"
        if needle not in text:
            raise RuntimeError(f"alloc-len state patch point not found: {path}")
        text = text.replace(
            needle,
            needle + "        self._mtp_side_alloc_lens = {}\n",
            1,
        )
        changed = True

    old_release = '''            seq_len = int(self._mtp_side_seq_lens.get(rid, 0))\n            row = req.req_pool_idx\n            if row is not None and seq_len > 0:\n                locs = self.req_to_token_pool.req_to_token[row, :seq_len].clone()\n'''
    new_release = '''            alloc_len = int(self._mtp_side_alloc_lens.get(rid, 0))\n            row = req.req_pool_idx\n            if row is not None and alloc_len > 0:\n                locs = self.req_to_token_pool.req_to_token[row, :alloc_len].clone()\n'''
    text, c = once(text, old_release, new_release, "release high-water", path)
    changed |= c

    if "self._mtp_side_alloc_lens.pop(rid, None)" not in text:
        needle = "            self._mtp_side_seq_lens.pop(rid, None)\n"
        if needle not in text:
            raise RuntimeError(f"release alloc-len pop point not found: {path}")
        text = text.replace(
            needle,
            needle + "            self._mtp_side_alloc_lens.pop(rid, None)\n",
            1,
        )
        changed = True

    if "self._mtp_side_alloc_lens[target_req.rid] = 0" not in text:
        needle = "        self._mtp_side_seq_lens[target_req.rid] = 0\n"
        if needle not in text:
            raise RuntimeError(f"reset alloc-len point not found: {path}")
        text = text.replace(
            needle,
            needle + "        self._mtp_side_alloc_lens[target_req.rid] = 0\n",
            1,
        )
        changed = True

    helper_anchor = "    def mtp_authoritative_prefill(\n"
    if "def _mtp_authoritative_reserve_span(" not in text:
        if helper_anchor not in text:
            raise RuntimeError(f"reserve helper anchor not found: {path}")
        text = text.replace(helper_anchor, HELPER + helper_anchor, 1)
        changed = True

    text, c = once(text, PREFILL_OLD, PREFILL_NEW, "prefill paged allocation", path)
    changed |= c
    text, c = once(text, DRAFT_OLD, DRAFT_NEW, "draft paged allocation", path)
    changed |= c
    text, c = once(text, EXTEND_OLD, EXTEND_NEW, "extend paged allocation", path)
    changed |= c
    text, c = once(text, TAIL_OLD, TAIL_NEW, "rejected-tail retention", path)
    changed |= c

    log_anchor = '''                logger.info(\n                    "[MTP-CUTOVER-PAGE] CUDA%d draft_page_size=%d allocator=%s",\n                    self.gpu_id,\n                    int(mr.page_size),\n                    type(mr.token_to_kv_pool_allocator).__name__,\n                )\n'''
    if PAGE64_MARKER not in text:
        if log_anchor not in text:
            raise RuntimeError(f"paged log anchor not found: {path}")
        text = text.replace(
            log_anchor,
            log_anchor
            + '''                logger.info(\n                    "[MTP-CUTOVER-PAGED] CUDA%d page_size=%d high-water allocator enabled",\n                    self.gpu_id,\n                    int(mr.page_size),\n                )\n''',
            1,
        )
        changed = True

    # Update stale page1 explanatory comments only; semantics are enforced above.
    text = text.replace(
        "keep the sidecar on the token allocator (page_size=1).",
        "keep the sidecar on a TRTLLM-compatible paged allocator (page_size=64).",
        1,
    )
    text = text.replace(
        "Otherwise PagedTokenToKVPoolAllocator.alloc() floors non-page-\n            # aligned sizes and silently returns too few slots.",
        "Arbitrary speculative spans are handled by the high-water alloc_extend path below.",
        1,
    )

    path.write_text(text)
    subprocess.run(["python3", "-m", "py_compile", str(path)], check=True)
    print(f"fixed CUDA2 page64 paged high-water allocation: {path}")
    return changed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--commit", action="store_true")
    args = ap.parse_args()

    host_changed = patch(HOST)
    fork_changed = patch(FORK)
    print(
        "MTP CUTOVER SIDECAR PAGE64 HOTFIX OK "
        f"host_changed={host_changed} fork_changed={fork_changed}"
    )

    if args.commit and fork_changed:
        subprocess.run(
            ["git", "add", str(FORK.relative_to(REPO))], cwd=REPO, check=True
        )
        diff = subprocess.run(
            ["git", "diff", "--cached", "--quiet"], cwd=REPO, check=False
        )
        if diff.returncode != 0:
            subprocess.run(
                [
                    "git",
                    "commit",
                    "-m",
                    "fix: make CUDA2 NVFP4 MTP sidecar page-aware",
                ],
                cwd=REPO,
                check=True,
            )
            subprocess.run(
                ["git", "push", "origin", "HEAD:wip/qwen38-mtp-sidecar-cuda2"],
                cwd=REPO,
                check=True,
            )
            print("MTP CUTOVER SIDECAR PAGE64 COMMIT/PUSH OK")


if __name__ == "__main__":
    main()
