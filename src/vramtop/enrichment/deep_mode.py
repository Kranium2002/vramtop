"""Deep mode scanner — reads memory stats from Unix sockets."""

from __future__ import annotations

import contextlib
import logging
import os
import socket
from pathlib import Path
from typing import Any

from vramtop.reporter.protocol import MemoryMsg, parse_message

logger = logging.getLogger(__name__)

_SOCKET_DIR = Path(os.environ.get("XDG_RUNTIME_DIR", "/tmp")) / "vramtop"  # nosec B108


def scan_sockets() -> list[Path]:
    """Find all .sock files in the socket dir owned by the current user."""
    if not _SOCKET_DIR.is_dir():
        return []

    result: list[Path] = []
    my_uid = os.getuid()

    for p in _SOCKET_DIR.glob("*.sock"):
        try:
            if os.stat(p).st_uid == my_uid:
                result.append(p)
        except (FileNotFoundError, PermissionError, OSError):
            continue

    return result


_MAX_READ_BYTES = 64 * 1024  # 64 KB cap to prevent memory pressure


def read_deep_data(sock_path: Path) -> dict[str, Any] | None:
    """Connect to a reporter socket and read the latest data."""

    conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    conn.settimeout(2.0)
    try:
        conn.connect(str(sock_path))
        data = b""
        while True:
            chunk = conn.recv(4096)
            if not chunk:
                break
            data += chunk
            if len(data) > _MAX_READ_BYTES:
                logger.warning(
                    "Deep mode socket %s exceeded %d byte cap",
                    sock_path, _MAX_READ_BYTES,
                )
                break
            # We expect handshake + memory on two lines
            if data.count(b"\n") >= 2:
                break
    except ConnectionRefusedError:
        # Stale socket — remove it
        with contextlib.suppress(OSError):
            sock_path.unlink(missing_ok=True)
        return None
    except (OSError, TimeoutError):
        return None
    finally:
        with contextlib.suppress(OSError):
            conn.close()

    lines = data.decode("utf-8", errors="replace").strip().splitlines()

    # Parse lines, looking for the last MemoryMsg
    result: dict[str, Any] = {}
    for line in lines:
        msg = parse_message(line)
        if isinstance(msg, MemoryMsg):
            result = {
                "deep_mode": True,
                "allocated_mb": msg.allocated_mb,
                "reserved_mb": msg.reserved_mb,
                "active_mb": msg.active_mb,
                "num_allocs": msg.num_allocs,
                "segments": msg.segments,
                "deep_ts": msg.ts,
            }

    return result if result else None


def get_deep_enrichment(pid: int) -> dict[str, Any] | None:
    """Get deep mode data for a specific PID.

    Handles Docker PID namespace mismatch: NVML reports host PIDs,
    but the reporter socket is named by the container PID.  When the
    exact PID socket doesn't exist, falls back to scanning all sockets.
    """
    sock_path = _SOCKET_DIR / f"{pid}.sock"
    if sock_path.exists():
        # Verify same UID
        try:
            if os.stat(sock_path).st_uid != os.getuid():
                return None
        except (FileNotFoundError, PermissionError, OSError):
            return None
        return read_deep_data(sock_path)

    # Docker PID namespace fallback: NVML host PID != container PID.
    # Scan all available sockets (already UID-checked by scan_sockets).
    sockets = scan_sockets()
    if not sockets:
        return None

    # Single socket: unambiguous match (common case).
    if len(sockets) == 1:
        return read_deep_data(sockets[0])

    # Multiple sockets: try each and return first success.
    # In practice there are rarely more than a few reporter sockets.
    for sock in sockets:
        data = read_deep_data(sock)
        if data is not None:
            return data

    return None
