"""NVIDIA MPS (Multi-Process Service) detection."""

from __future__ import annotations

import time

import psutil  # type: ignore[import-untyped]

from vramtop.permissions import is_same_user

_MPS_DAEMON_NAME = "nvidia-cuda-mps-control"

# Cache: (result, timestamp)
_mps_cache: tuple[bool, float] | None = None
_CACHE_TTL = 30.0


def is_mps_client(pid: int) -> bool:
    """Check if a process might be an MPS client.

    A process is considered an MPS client if the MPS daemon is running
    and the process belongs to the same user.
    """
    if not is_same_user(pid):
        return False
    return detect_mps_daemon()


def detect_mps_daemon() -> bool:
    """Detect if nvidia-cuda-mps-control is running. Cached for 30s."""
    global _mps_cache

    now = time.monotonic()
    if _mps_cache is not None:
        result, ts = _mps_cache
        if now - ts < _CACHE_TTL:
            return result

    found = _scan_for_mps()
    _mps_cache = (found, now)
    return found


def _scan_for_mps() -> bool:
    """Scan process list for MPS control daemon."""
    try:
        for proc in psutil.process_iter(["name"]):
            try:
                if proc.info["name"] == _MPS_DAEMON_NAME:
                    return True
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
    except Exception:
        pass
    return False


def _reset_cache() -> None:
    """Reset the MPS cache (for testing)."""
    global _mps_cache
    _mps_cache = None
