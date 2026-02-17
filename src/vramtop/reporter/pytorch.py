"""PyTorch deep mode reporter — sends memory stats over Unix socket."""

from __future__ import annotations

import atexit
import contextlib
import logging
import os
import socket
import threading
import time
from pathlib import Path

from vramtop.reporter.protocol import (
    PROTOCOL_VERSION,
    HandshakeMsg,
    MemoryMsg,
    to_json,
)

logger = logging.getLogger(__name__)

_SOCKET_DIR = Path(os.environ.get("XDG_RUNTIME_DIR", "/tmp")) / "vramtop"  # nosec B108

_active = False
_socket_path: Path | None = None
_lock = threading.Lock()


def _get_memory_stats(cuda_device: int) -> MemoryMsg:
    """Read PyTorch CUDA memory stats, fallback to zeros."""
    try:
        import torch

        stats = torch.cuda.memory_stats(cuda_device)
        allocated = stats.get("allocated_bytes.all.current", 0) / (1024 * 1024)
        reserved = stats.get("reserved_bytes.all.current", 0) / (1024 * 1024)
        active = stats.get("active_bytes.all.current", 0) / (1024 * 1024)
        num_allocs = stats.get("allocation.all.current", 0)
        segments = stats.get("segment.all.current", 0)
    except Exception:
        allocated = 0.0
        reserved = 0.0
        active = 0.0
        num_allocs = 0
        segments = 0

    return MemoryMsg(
        ts=time.time(),
        allocated_mb=allocated,
        reserved_mb=reserved,
        active_mb=active,
        num_allocs=num_allocs,
        segments=segments,
    )


def _reporter_thread(
    framework: str, cuda_device: int, interval: float
) -> None:
    """Daemon thread that sends memory stats over Unix socket."""
    global _socket_path  # noqa: PLW0603

    try:
        _SOCKET_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)

        sock_path = _SOCKET_DIR / f"{os.getpid()}.sock"
        _socket_path = sock_path

        # Clean up stale socket
        if sock_path.exists():
            sock_path.unlink()

        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(str(sock_path))
        server.listen(1)
        server.settimeout(interval)

        handshake = HandshakeMsg(
            v=PROTOCOL_VERSION,
            pid=os.getpid(),
            framework=framework,
            cuda_device=cuda_device,
        )

        while _active:
            # Accept connection or timeout
            try:
                conn, _ = server.accept()
            except TimeoutError:
                continue
            except OSError:
                break

            try:
                conn.settimeout(2.0)
                # Send handshake
                conn.sendall((to_json(handshake) + "\n").encode())
                # Send latest memory stats
                msg = _get_memory_stats(cuda_device)
                conn.sendall((to_json(msg) + "\n").encode())
            except OSError:
                pass
            finally:
                with contextlib.suppress(OSError):
                    conn.close()

        server.close()
    except Exception:
        logger.debug("Reporter thread failed", exc_info=True)


def _cleanup() -> None:
    """Remove socket file on exit."""
    global _active, _socket_path  # noqa: PLW0603

    _active = False
    if _socket_path is not None:
        with contextlib.suppress(OSError):
            _socket_path.unlink(missing_ok=True)
        _socket_path = None


def report(
    framework: str = "pytorch", cuda_device: int = 0, interval: float = 1.0
) -> None:
    """Start the deep mode reporter (idempotent).

    Usage::

        import vramtop
        vramtop.report()
    """
    global _active  # noqa: PLW0603

    with _lock:
        if _active:
            return
        _active = True

    atexit.register(_cleanup)

    t = threading.Thread(
        target=_reporter_thread,
        args=(framework, cuda_device, interval),
        daemon=True,
        name="vramtop-reporter",
    )
    t.start()
