"""Tests for Docker PID namespace resolution in enrichment."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from vramtop.enrichment import _resolve_pid, _scan_container_gpu_pids


@pytest.fixture(autouse=True)
def _clear_cache() -> None:
    """Reset the GPU PID scan cache between tests."""
    import vramtop.enrichment as mod

    mod._container_gpu_pids = []
    mod._container_gpu_pids_time = 0.0
    yield
    mod._container_gpu_pids = []
    mod._container_gpu_pids_time = 0.0


class TestResolvePid:
    """Test _resolve_pid for Docker PID namespace mapping."""

    def test_pid_exists_in_proc(self, tmp_path: Path) -> None:
        """If /proc/{pid} exists, return it unchanged."""
        # os.getpid() always exists in /proc/
        result = _resolve_pid(os.getpid())
        assert result == os.getpid()

    @patch("vramtop.enrichment.os.path.exists")
    @patch("vramtop.enrichment._scan_container_gpu_pids")
    def test_phantom_pid_single_gpu_process(
        self, mock_scan: MagicMock, mock_exists: MagicMock
    ) -> None:
        """Phantom PID with one GPU container process → map to it."""
        mock_exists.return_value = False
        mock_scan.return_value = [42]

        result = _resolve_pid(999999)
        assert result == 42

    @patch("vramtop.enrichment.os.path.exists")
    @patch("vramtop.enrichment._scan_container_gpu_pids")
    def test_phantom_pid_no_gpu_processes(
        self, mock_scan: MagicMock, mock_exists: MagicMock
    ) -> None:
        """Phantom PID with no GPU processes → return original."""
        mock_exists.return_value = False
        mock_scan.return_value = []

        result = _resolve_pid(999999)
        assert result == 999999

    @patch("vramtop.enrichment.os.path.exists")
    @patch("vramtop.enrichment._scan_container_gpu_pids")
    def test_phantom_pid_multiple_gpu_processes(
        self, mock_scan: MagicMock, mock_exists: MagicMock
    ) -> None:
        """Phantom PID with multiple GPU processes → return first non-vramtop."""
        mock_exists.return_value = False
        mock_scan.return_value = [100, 200, 300]

        # Mock /proc/{pid}/comm reads: first is vramtop, second is python
        def fake_open(path: str, *args: object, **kwargs: object) -> MagicMock:
            m = MagicMock()
            if "/100/" in path:
                m.__enter__ = lambda s: MagicMock(read=lambda sz: b"vramtop\n")
            elif "/200/" in path:
                m.__enter__ = lambda s: MagicMock(read=lambda sz: b"python\n")
            else:
                m.__enter__ = lambda s: MagicMock(read=lambda sz: b"other\n")
            m.__exit__ = lambda s, *a: None
            return m

        with patch("builtins.open", side_effect=fake_open):
            result = _resolve_pid(999999)
        assert result == 200  # First non-vramtop


class TestDeepModeFallback:
    """Test deep mode socket scanning fallback for Docker."""

    def test_exact_pid_match(self, tmp_path: Path) -> None:
        """When socket exists for exact PID, use it."""
        from vramtop.enrichment.deep_mode import get_deep_enrichment

        # No socket dir → returns None
        with patch("vramtop.enrichment.deep_mode._SOCKET_DIR", tmp_path / "nosuch"):
            result = get_deep_enrichment(12345)
        assert result is None

    @patch("vramtop.enrichment.deep_mode.scan_sockets")
    @patch("vramtop.enrichment.deep_mode.read_deep_data")
    def test_fallback_single_socket(
        self, mock_read: MagicMock, mock_scan: MagicMock, tmp_path: Path
    ) -> None:
        """When exact PID socket missing, fall back to scanning."""
        from vramtop.enrichment.deep_mode import get_deep_enrichment

        mock_scan.return_value = [tmp_path / "42.sock"]
        mock_read.return_value = {
            "deep_mode": True,
            "allocated_mb": 100.0,
            "reserved_mb": 200.0,
            "active_mb": 100.0,
            "num_allocs": 10,
            "segments": 5,
            "deep_ts": 1234567890.0,
        }

        with patch("vramtop.enrichment.deep_mode._SOCKET_DIR", tmp_path):
            result = get_deep_enrichment(999999)

        assert result is not None
        assert result["deep_mode"] is True
        assert result["allocated_mb"] == 100.0

    @patch("vramtop.enrichment.deep_mode.scan_sockets")
    def test_fallback_no_sockets(self, mock_scan: MagicMock, tmp_path: Path) -> None:
        """When no sockets available, return None."""
        from vramtop.enrichment.deep_mode import get_deep_enrichment

        mock_scan.return_value = []

        with patch("vramtop.enrichment.deep_mode._SOCKET_DIR", tmp_path):
            result = get_deep_enrichment(999999)

        assert result is None


class TestEnrichProcessDockerIntegration:
    """Test the full enrich_process flow with Docker PID namespace."""

    @patch("vramtop.enrichment._resolve_pid")
    @patch("vramtop.enrichment.is_same_user")
    def test_enrichment_uses_resolved_pid(
        self, mock_same_user: MagicMock, mock_resolve: MagicMock
    ) -> None:
        """enrich_process uses resolved PID for is_same_user check."""
        mock_resolve.return_value = 42
        mock_same_user.return_value = False

        from vramtop.enrichment import enrich_process

        result = enrich_process(999999, 0)
        mock_resolve.assert_called_once_with(999999)
        mock_same_user.assert_called_once_with(42)
        assert result.framework is None  # Enrichment skipped (same_user=False)
