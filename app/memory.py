"""Process memory hygiene between jobs.

Prod RSS climbed ~3.5 -> 11 GB over a couple of hundred sequential jobs and was
OOMKilled mid-job at the 12Gi limit (2026-09-24), with each film's own peak
bounded by chunked ASR. The cause wasn't reproducible locally (varied films on a
worker thread settled at ~3.2 GB, or ~2.2 GB with malloc_trim), which points at
glibc heap fragmentation from large short-lived numpy/torch buffers rather than a
Python-level leak. So, in layers: hand freed heap back to the OS after every
job, cap glibc arenas in the image (MALLOC_ARENA_MAX), and if RSS still ends a
job above a fraction of the container limit, restart cleanly between jobs
instead of being killed during one.
"""

from __future__ import annotations

import ctypes
import gc
import logging
from pathlib import Path

log = logging.getLogger(__name__)

_CGROUP_V2 = Path("/sys/fs/cgroup/memory.max")
_CGROUP_V1 = Path("/sys/fs/cgroup/memory/memory.limit_in_bytes")
# cgroup v1 reports "no limit" as a huge number rather than "max".
_NO_LIMIT = 1 << 60


def rss_bytes() -> int:
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return 0


def limit_bytes() -> int:
    """The container's memory limit, or 0 when there is none (or no cgroup)."""
    for path in (_CGROUP_V2, _CGROUP_V1):
        try:
            raw = path.read_text().strip()
        except OSError:
            continue
        if raw == "max":
            return 0
        try:
            value = int(raw)
        except ValueError:
            return 0
        return 0 if value >= _NO_LIMIT else value
    return 0


def release() -> None:
    """Collect cycles, then return freed heap pages to the OS (glibc only)."""
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass  # not glibc: nothing to trim


def over_budget(fraction: float) -> tuple[bool, int, int]:
    """(rss > fraction * limit, rss, limit). Never true without a known limit."""
    rss, limit = rss_bytes(), limit_bytes()
    return bool(fraction > 0 and limit and rss > fraction * limit), rss, limit
