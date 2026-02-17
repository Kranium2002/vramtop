"""Framework detection from /proc/<pid>/cmdline and /proc/<pid>/maps."""

from __future__ import annotations

import re
import time

from vramtop.permissions import is_same_user

# Cache: (pid, starttime) -> (framework, version, timestamp)
_cache: dict[tuple[int, int], tuple[str | None, str | None, float]] = {}
_CACHE_TTL = 30.0

# Cmdline patterns: (regex, framework_name)
_CMDLINE_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"ollama"), "ollama"),
    (re.compile(r"vllm"), "vllm"),
    (re.compile(r"sglang"), "sglang"),
    (re.compile(r"text-generation-launcher|text-generation-server"), "tgi"),
    (re.compile(r"llama[-_]?cpp|llama[-_]?server|main\b.*--model"), "llamacpp"),
]

# Maps patterns: (library regex, framework_name)
_MAPS_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"libtorch"), "pytorch"),
    (re.compile(r"libjax"), "jax"),
    (re.compile(r"libtensorflow"), "tensorflow"),
]


def detect_framework(
    pid: int, starttime: int = 0
) -> tuple[str | None, str | None]:
    """Detect the ML framework used by a process.

    Returns (framework_name, version) or (None, None).
    Requires same-UID. Results cached by (pid, starttime) with 30s TTL.
    """
    now = time.monotonic()
    key = (pid, starttime)

    cached = _cache.get(key)
    if cached is not None:
        fw, ver, ts = cached
        if now - ts < _CACHE_TTL:
            return fw, ver

    if not is_same_user(pid):
        return None, None

    fw, ver = _detect_from_cmdline(pid)
    if fw is None:
        fw, ver = _detect_from_maps(pid)

    _cache[key] = (fw, ver, now)
    return fw, ver


def _detect_from_cmdline(pid: int) -> tuple[str | None, str | None]:
    """Check /proc/<pid>/cmdline for known framework patterns."""
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            raw = f.read(4096)
        cmdline = raw.replace(b"\x00", b" ").decode("utf-8", errors="replace")
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return None, None

    for pattern, name in _CMDLINE_PATTERNS:
        if pattern.search(cmdline):
            return name, None

    return None, None


def _detect_from_maps(pid: int) -> tuple[str | None, str | None]:
    """Check /proc/<pid>/maps for loaded shared libraries."""
    try:
        with open(f"/proc/{pid}/maps") as f:
            raw = f.read(256 * 1024)  # Cap at 256KB
        maps_content = raw if isinstance(raw, str) else raw.decode("utf-8", errors="replace")
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return None, None

    for pattern, name in _MAPS_PATTERNS:
        if pattern.search(maps_content):
            return name, None

    return None, None
