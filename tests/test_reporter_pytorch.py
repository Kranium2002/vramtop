"""Unit tests for the PyTorch deep mode reporter."""

from __future__ import annotations

import os
import socket
import threading
import time
from typing import TYPE_CHECKING
from unittest.mock import patch

if TYPE_CHECKING:
    from pathlib import Path

from vramtop.reporter.protocol import HandshakeMsg, MemoryMsg, parse_message


class TestReporterCreatesSocket:
    def test_report_creates_socket_file(self, tmp_path: Path) -> None:
        from vramtop.reporter import pytorch

        sock_dir = tmp_path / "vramtop"

        with patch.object(pytorch, "_SOCKET_DIR", sock_dir), \
             patch.object(pytorch, "_active", False), \
             patch.object(pytorch, "_socket_path", None):
            pytorch._active = False

            # Patch _lock to a fresh lock
            with patch.object(pytorch, "_lock", threading.Lock()):
                pytorch.report(framework="pytorch", cuda_device=0, interval=0.5)

            # Give daemon thread time to start
            time.sleep(0.5)

            sock_path = sock_dir / f"{os.getpid()}.sock"
            assert sock_path.exists()

            # Cleanup
            pytorch._active = False
            time.sleep(0.5)
            if sock_path.exists():
                sock_path.unlink()

    def test_report_is_idempotent(self, tmp_path: Path) -> None:
        from vramtop.reporter import pytorch

        sock_dir = tmp_path / "vramtop"
        call_count = 0
        original_thread_init = threading.Thread.__init__

        def counting_init(self: threading.Thread, *args: object, **kwargs: object) -> None:
            nonlocal call_count
            if kwargs.get("name") == "vramtop-reporter":
                call_count += 1
            original_thread_init(self, *args, **kwargs)  # type: ignore[arg-type]

        with patch.object(pytorch, "_SOCKET_DIR", sock_dir), \
             patch.object(pytorch, "_active", False), \
             patch.object(pytorch, "_socket_path", None), \
             patch.object(pytorch, "_lock", threading.Lock()):
            pytorch.report(framework="pytorch", cuda_device=0, interval=0.5)
            # Second call should be no-op since _active is True
            pytorch.report(framework="pytorch", cuda_device=0, interval=0.5)

            # Give time, cleanup
            time.sleep(0.3)
            pytorch._active = False
            time.sleep(0.5)

    def test_daemon_thread_is_daemon(self, tmp_path: Path) -> None:
        from vramtop.reporter import pytorch

        sock_dir = tmp_path / "vramtop"
        created_threads: list[threading.Thread] = []
        original_start = threading.Thread.start

        def capture_start(self: threading.Thread) -> None:
            if self.name == "vramtop-reporter":
                created_threads.append(self)
            original_start(self)

        with patch.object(pytorch, "_SOCKET_DIR", sock_dir), \
             patch.object(pytorch, "_active", False), \
             patch.object(pytorch, "_socket_path", None), \
             patch.object(pytorch, "_lock", threading.Lock()), \
             patch.object(threading.Thread, "start", capture_start):
            pytorch.report(framework="pytorch", cuda_device=0, interval=0.5)

            time.sleep(0.3)
            assert len(created_threads) == 1
            assert created_threads[0].daemon is True

            pytorch._active = False
            time.sleep(0.5)


class TestHandshakeFormat:
    def test_handshake_sent_on_connect(self, tmp_path: Path) -> None:
        from vramtop.reporter import pytorch

        sock_dir = tmp_path / "vramtop"

        with patch.object(pytorch, "_SOCKET_DIR", sock_dir), \
             patch.object(pytorch, "_active", False), \
             patch.object(pytorch, "_socket_path", None), \
             patch.object(pytorch, "_lock", threading.Lock()):
            pytorch.report(framework="pytorch", cuda_device=0, interval=0.5)

            time.sleep(0.5)

            sock_path = sock_dir / f"{os.getpid()}.sock"
            assert sock_path.exists()

            # Connect and read
            client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            client.settimeout(3.0)
            client.connect(str(sock_path))

            data = b""
            while data.count(b"\n") < 2:
                chunk = client.recv(4096)
                if not chunk:
                    break
                data += chunk

            client.close()

            lines = data.decode().strip().splitlines()
            assert len(lines) >= 2

            handshake = parse_message(lines[0])
            assert isinstance(handshake, HandshakeMsg)
            assert handshake.v == 1
            assert handshake.pid == os.getpid()
            assert handshake.framework == "pytorch"
            assert handshake.cuda_device == 0

            memory = parse_message(lines[1])
            assert isinstance(memory, MemoryMsg)

            pytorch._active = False
            time.sleep(0.5)


class TestCleanup:
    def test_cleanup_removes_socket(self, tmp_path: Path) -> None:
        from vramtop.reporter import pytorch

        sock_dir = tmp_path / "vramtop"

        with patch.object(pytorch, "_SOCKET_DIR", sock_dir), \
             patch.object(pytorch, "_active", False), \
             patch.object(pytorch, "_socket_path", None), \
             patch.object(pytorch, "_lock", threading.Lock()):
            pytorch.report(framework="pytorch", cuda_device=0, interval=0.5)

            time.sleep(0.5)

            sock_path = sock_dir / f"{os.getpid()}.sock"
            assert sock_path.exists()

            # Call cleanup
            pytorch._cleanup()
            assert not sock_path.exists()
            assert pytorch._active is False


class TestSocketDirPermissions:
    def test_socket_dir_has_0o700(self, tmp_path: Path) -> None:
        from vramtop.reporter import pytorch

        sock_dir = tmp_path / "vramtop"

        with patch.object(pytorch, "_SOCKET_DIR", sock_dir), \
             patch.object(pytorch, "_active", False), \
             patch.object(pytorch, "_socket_path", None), \
             patch.object(pytorch, "_lock", threading.Lock()):
            pytorch.report(framework="pytorch", cuda_device=0, interval=0.5)

            time.sleep(0.5)

            mode = sock_dir.stat().st_mode & 0o777
            assert mode == 0o700

            pytorch._active = False
            time.sleep(0.5)
