"""Unit tests for deep mode protocol and scanner."""

from __future__ import annotations

import json
import os
import socket
import threading
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

if TYPE_CHECKING:
    from pathlib import Path

from vramtop.reporter.protocol import (
    HandshakeMsg,
    MemoryMsg,
    parse_message,
    to_json,
)


class TestHandshakeMsg:
    def test_to_json_roundtrip(self) -> None:
        msg = HandshakeMsg(v=1, pid=1234, framework="pytorch", cuda_device=0)
        line = to_json(msg)
        parsed = parse_message(line)
        assert isinstance(parsed, HandshakeMsg)
        assert parsed.v == 1
        assert parsed.pid == 1234
        assert parsed.framework == "pytorch"
        assert parsed.cuda_device == 0

    def test_to_json_contains_type(self) -> None:
        msg = HandshakeMsg(v=1, pid=1, framework="jax", cuda_device=2)
        data = json.loads(to_json(msg))
        assert data["type"] == "handshake"


class TestMemoryMsg:
    def test_to_json_roundtrip(self) -> None:
        msg = MemoryMsg(
            ts=1000.0,
            allocated_mb=512.5,
            reserved_mb=1024.0,
            active_mb=256.0,
            num_allocs=100,
            segments=10,
        )
        line = to_json(msg)
        parsed = parse_message(line)
        assert isinstance(parsed, MemoryMsg)
        assert parsed.ts == 1000.0
        assert parsed.allocated_mb == 512.5
        assert parsed.reserved_mb == 1024.0
        assert parsed.active_mb == 256.0
        assert parsed.num_allocs == 100
        assert parsed.segments == 10

    def test_to_json_contains_type(self) -> None:
        msg = MemoryMsg(
            ts=0.0, allocated_mb=0.0, reserved_mb=0.0,
            active_mb=0.0, num_allocs=0, segments=0,
        )
        data = json.loads(to_json(msg))
        assert data["type"] == "memory"


class TestParseMessage:
    def test_invalid_json_returns_none(self) -> None:
        assert parse_message("not json") is None

    def test_empty_string_returns_none(self) -> None:
        assert parse_message("") is None

    def test_missing_type_returns_none(self) -> None:
        assert parse_message('{"v": 1}') is None

    def test_unknown_type_returns_none(self) -> None:
        assert parse_message('{"type": "unknown"}') is None

    def test_missing_fields_returns_none(self) -> None:
        assert parse_message('{"type": "handshake", "v": 1}') is None

    def test_non_dict_returns_none(self) -> None:
        assert parse_message("[1, 2, 3]") is None


class TestScanSockets:
    def test_finds_sock_files(self, tmp_path: Path) -> None:
        from vramtop.enrichment import deep_mode

        sock_dir = tmp_path / "vramtop"
        sock_dir.mkdir()
        (sock_dir / "123.sock").touch()
        (sock_dir / "456.sock").touch()

        with patch.object(deep_mode, "_SOCKET_DIR", sock_dir):
            result = deep_mode.scan_sockets()

        assert len(result) == 2
        names = {p.name for p in result}
        assert names == {"123.sock", "456.sock"}

    def test_skips_non_sock_files(self, tmp_path: Path) -> None:
        from vramtop.enrichment import deep_mode

        sock_dir = tmp_path / "vramtop"
        sock_dir.mkdir()
        (sock_dir / "123.sock").touch()
        (sock_dir / "data.txt").touch()
        (sock_dir / "lock.pid").touch()

        with patch.object(deep_mode, "_SOCKET_DIR", sock_dir):
            result = deep_mode.scan_sockets()

        assert len(result) == 1
        assert result[0].name == "123.sock"

    def test_empty_dir(self, tmp_path: Path) -> None:
        from vramtop.enrichment import deep_mode

        sock_dir = tmp_path / "vramtop"
        sock_dir.mkdir()

        with patch.object(deep_mode, "_SOCKET_DIR", sock_dir):
            result = deep_mode.scan_sockets()

        assert result == []

    def test_missing_dir(self, tmp_path: Path) -> None:
        from vramtop.enrichment import deep_mode

        sock_dir = tmp_path / "nonexistent"

        with patch.object(deep_mode, "_SOCKET_DIR", sock_dir):
            result = deep_mode.scan_sockets()

        assert result == []


class TestStaleSocketHandling:
    def test_stale_socket_removed(self, tmp_path: Path) -> None:
        from vramtop.enrichment.deep_mode import read_deep_data

        sock_path = tmp_path / "stale.sock"
        sock_path.touch()

        result = read_deep_data(sock_path)
        assert result is None


class TestGetDeepEnrichment:
    def test_matching_pid_with_live_socket(self, tmp_path: Path) -> None:
        from vramtop.enrichment import deep_mode

        sock_dir = tmp_path / "vramtop"
        sock_dir.mkdir()

        sock_path = sock_dir / "99999.sock"

        # Set up a real Unix domain socket server
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(str(sock_path))
        server.listen(1)

        handshake = to_json(
            HandshakeMsg(v=1, pid=99999, framework="pytorch", cuda_device=0)
        )
        mem = to_json(
            MemoryMsg(
                ts=1000.0,
                allocated_mb=512.0,
                reserved_mb=1024.0,
                active_mb=256.0,
                num_allocs=50,
                segments=5,
            )
        )

        def serve() -> None:
            conn, _ = server.accept()
            conn.sendall(f"{handshake}\n{mem}\n".encode())
            conn.close()

        t = threading.Thread(target=serve, daemon=True)
        t.start()

        with patch.object(deep_mode, "_SOCKET_DIR", sock_dir):
            result = deep_mode.get_deep_enrichment(99999)

        server.close()
        t.join(timeout=2)

        assert result is not None
        assert result["deep_mode"] is True
        assert result["allocated_mb"] == 512.0
        assert result["reserved_mb"] == 1024.0
        assert result["num_allocs"] == 50

    def test_no_socket_returns_none(self, tmp_path: Path) -> None:
        from vramtop.enrichment import deep_mode

        sock_dir = tmp_path / "vramtop"
        sock_dir.mkdir()

        with patch.object(deep_mode, "_SOCKET_DIR", sock_dir):
            result = deep_mode.get_deep_enrichment(12345)

        assert result is None


class TestPermissionCheck:
    def test_skips_different_uid(self, tmp_path: Path) -> None:
        from vramtop.enrichment import deep_mode

        sock_dir = tmp_path / "vramtop"
        sock_dir.mkdir()
        sock_path = sock_dir / "123.sock"
        sock_path.touch()

        # Mock os.stat to return a different UID
        original_stat = os.stat

        def fake_stat(path: str | Path, **kwargs: Any) -> os.stat_result:
            s = original_stat(path, **kwargs)
            if str(path) == str(sock_path):
                # Create a fake stat result with different UID
                # We can't easily change UID, so mock getuid instead
                return s
            return s

        with patch.object(deep_mode, "_SOCKET_DIR", sock_dir), \
             patch("vramtop.enrichment.deep_mode.os.getuid", return_value=99999):
            result = deep_mode.get_deep_enrichment(123)

        assert result is None
