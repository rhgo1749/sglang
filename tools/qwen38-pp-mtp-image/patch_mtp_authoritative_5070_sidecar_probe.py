#!/usr/bin/env python3
"""Diagnostic-only placeholder for authoritative MTP placement work.

The current fork contains CUDA2/sidecar primitives in eagle_worker_v2.py, but
production native-MTP remains colocated with PP-last. Do not silently reroute the
authoritative draft worker until the scheduler/PP bridge cutover contract is
validated end-to-end. This installer exists only to fail loudly if someone tries
to treat device remapping as a valid placement experiment.
"""
raise RuntimeError(
    "authoritative MTP 5070-sidecar cutover is not yet wired; "
    "do not benchmark placement by CUDA_VISIBLE_DEVICES remapping"
)
