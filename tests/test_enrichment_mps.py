"""Tests for NVIDIA MPS daemon detection."""

from __future__ import annotations

from unittest.mock import patch, MagicMock

import pytest

from vramtop.enrichment.mps import (
    detect_mps_daemon,
    is_mps_client,
    _reset_cache,
)


@pytest.fixture(autouse=True)
def _clear_cache():
    """Reset MPS cache before each test."""
    _reset_cache()
    yield
    _reset_cache()


class TestDetectMpsDaemon:
    """Test MPS daemon detection via process scan."""

    def test_daemon_found(self):
        proc = MagicMock()
        proc.info = {"name": "nvidia-cuda-mps-control"}

        with patch("psutil.process_iter", return_value=[proc]):
            assert detect_mps_daemon() is True

    def test_daemon_not_found(self):
        proc = MagicMock()
        proc.info = {"name": "python3"}

        with patch("psutil.process_iter", return_value=[proc]):
            assert detect_mps_daemon() is False

    def test_empty_process_list(self):
        with patch("psutil.process_iter", return_value=[]):
            assert detect_mps_daemon() is False

    def test_result_cached(self):
        proc = MagicMock()
        proc.info = {"name": "nvidia-cuda-mps-control"}

        with patch("psutil.process_iter", return_value=[proc]):
            r1 = detect_mps_daemon()

        # Second call should use cache
        with patch("psutil.process_iter", side_effect=AssertionError("should not call")):
            r2 = detect_mps_daemon()

        assert r1 == r2 is True

    def test_cache_expires(self):
        import vramtop.enrichment.mps as mps_mod

        proc_mps = MagicMock()
        proc_mps.info = {"name": "nvidia-cuda-mps-control"}

        with patch("psutil.process_iter", return_value=[proc_mps]):
            detect_mps_daemon()

        # Expire the cache manually
        assert mps_mod._mps_cache is not None
        result, ts = mps_mod._mps_cache
        mps_mod._mps_cache = (result, ts - mps_mod._CACHE_TTL - 1)

        proc_other = MagicMock()
        proc_other.info = {"name": "python3"}

        with patch("psutil.process_iter", return_value=[proc_other]):
            r2 = detect_mps_daemon()

        assert r2 is False


class TestIsMpsClient:
    """Test per-process MPS client detection."""

    @patch("vramtop.enrichment.mps.is_same_user", return_value=True)
    def test_mps_client_when_daemon_running(self, _mock_uid):
        proc = MagicMock()
        proc.info = {"name": "nvidia-cuda-mps-control"}

        with patch("psutil.process_iter", return_value=[proc]):
            assert is_mps_client(1234) is True

    @patch("vramtop.enrichment.mps.is_same_user", return_value=True)
    def test_not_mps_client_when_no_daemon(self, _mock_uid):
        with patch("psutil.process_iter", return_value=[]):
            assert is_mps_client(1234) is False

    @patch("vramtop.enrichment.mps.is_same_user", return_value=False)
    def test_same_uid_enforced(self, _mock_uid):
        assert is_mps_client(1234) is False
