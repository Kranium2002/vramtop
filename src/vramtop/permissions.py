"""UID-based permission checks for /proc access."""

from __future__ import annotations

import os


def is_same_user(pid: int) -> bool:
    """Check if a process is owned by the current user."""
    try:
        return os.stat(f"/proc/{pid}").st_uid == os.getuid()
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return False


def check_proc_readable(pid: int) -> bool:
    """Check if /proc/<pid> is accessible."""
    try:
        os.listdir(f"/proc/{pid}")
        return True
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return False
