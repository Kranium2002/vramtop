"""Tests for framework detection from cmdline and maps."""

from __future__ import annotations

from unittest.mock import mock_open, patch

import pytest

from vramtop.enrichment.detector import (
    _cache,
    _CACHE_TTL,
    detect_framework,
)


@pytest.fixture(autouse=True)
def _clear_cache():
    """Clear the detector cache before each test."""
    _cache.clear()
    yield
    _cache.clear()


class TestCmdlineDetection:
    """Test detection from /proc/<pid>/cmdline."""

    @pytest.mark.parametrize(
        "cmdline,expected_framework",
        [
            (b"ollama\x00serve", "ollama"),
            (b"/usr/bin/vllm\x00--model\x00foo", "vllm"),
            (b"python\x00-m\x00sglang.serve", "sglang"),
            (b"text-generation-launcher\x00--model\x00x", "tgi"),
            (b"text-generation-server\x00run", "tgi"),
            (b"llama-server\x00--model\x00foo.gguf", "llamacpp"),
            (b"llama_cpp\x00serve", "llamacpp"),
            (b"llamacpp\x00serve", "llamacpp"),
            (b"/usr/bin/python3\x00train.py", None),
        ],
    )
    @patch("vramtop.enrichment.detector.is_same_user", return_value=True)
    def test_cmdline_patterns(self, _mock_uid, cmdline, expected_framework):
        m_cmdline = mock_open(read_data=cmdline)
        # Maps file with no framework libraries (for fallthrough case)
        m_maps = mock_open(read_data="7f00 r-xp 00000 /usr/lib/libc.so.6\n")

        def open_side_effect(path, *args, **kwargs):
            if "cmdline" in str(path):
                return m_cmdline()
            if "maps" in str(path):
                return m_maps()
            raise FileNotFoundError(path)

        with patch("builtins.open", side_effect=open_side_effect):
            fw, ver = detect_framework(1234, starttime=1)
        assert fw == expected_framework
        assert ver is None

    @patch("vramtop.enrichment.detector.is_same_user", return_value=False)
    def test_same_uid_enforced(self, _mock_uid):
        fw, ver = detect_framework(1234, starttime=1)
        assert fw is None
        assert ver is None


class TestMapsDetection:
    """Test detection from /proc/<pid>/maps."""

    @pytest.mark.parametrize(
        "maps_content,expected_framework",
        [
            ("7f00 r-xp 00000 /usr/lib/libtorch.so\n", "pytorch"),
            ("7f00 r-xp 00000 /usr/lib/libjax.so.1\n", "jax"),
            ("7f00 r-xp 00000 /usr/lib/libtensorflow.so.2\n", "tensorflow"),
            ("7f00 r-xp 00000 /usr/lib/libc.so.6\n", None),
        ],
    )
    @patch("vramtop.enrichment.detector.is_same_user", return_value=True)
    def test_maps_patterns(self, _mock_uid, maps_content, expected_framework):
        # cmdline returns nothing recognizable, falls through to maps
        cmdline_data = b"python3\x00train.py"
        m_cmdline = mock_open(read_data=cmdline_data)
        m_maps = mock_open(read_data=maps_content)

        def open_side_effect(path, *args, **kwargs):
            if "cmdline" in str(path):
                return m_cmdline()
            if "maps" in str(path):
                return m_maps()
            raise FileNotFoundError(path)

        with patch("builtins.open", side_effect=open_side_effect):
            fw, ver = detect_framework(9999, starttime=2)
        assert fw == expected_framework


class TestCacheTTL:
    """Test cache behavior."""

    @patch("vramtop.enrichment.detector.is_same_user", return_value=True)
    def test_cache_hit(self, _mock_uid):
        cmdline_data = b"ollama\x00serve"
        with patch("builtins.open", mock_open(read_data=cmdline_data)):
            fw1, _ = detect_framework(100, starttime=5)

        # Second call should use cache (no file reads needed)
        with patch("builtins.open", side_effect=AssertionError("should not read")):
            fw2, _ = detect_framework(100, starttime=5)

        assert fw1 == fw2 == "ollama"

    @patch("vramtop.enrichment.detector.is_same_user", return_value=True)
    def test_cache_expires(self, _mock_uid):
        import time as _time

        cmdline_data = b"ollama\x00serve"
        with patch("builtins.open", mock_open(read_data=cmdline_data)):
            detect_framework(100, starttime=5)

        # Expire the cache
        key = (100, 5)
        fw, ver, ts = _cache[key]
        _cache[key] = (fw, ver, ts - _CACHE_TTL - 1)

        # Now it should re-read
        new_data = b"vllm\x00serve"
        with patch("builtins.open", mock_open(read_data=new_data)):
            fw2, _ = detect_framework(100, starttime=5)
        assert fw2 == "vllm"

    @patch("vramtop.enrichment.detector.is_same_user", return_value=True)
    def test_different_starttime_different_cache(self, _mock_uid):
        cmdline_data = b"ollama\x00serve"
        with patch("builtins.open", mock_open(read_data=cmdline_data)):
            fw1, _ = detect_framework(100, starttime=5)

        new_data = b"vllm\x00serve"
        with patch("builtins.open", mock_open(read_data=new_data)):
            fw2, _ = detect_framework(100, starttime=6)

        assert fw1 == "ollama"
        assert fw2 == "vllm"
