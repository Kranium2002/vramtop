"""Tests for model file scanning via /proc/<pid>/fd/."""

from __future__ import annotations

from unittest.mock import patch, MagicMock

import pytest

from vramtop.enrichment.model_files import scan_model_files, _MAX_FDS


class TestScanModelFiles:
    """Test model file detection."""

    @patch("vramtop.enrichment.model_files.is_same_user", return_value=False)
    def test_same_uid_enforced(self, _mock_uid):
        result = scan_model_files(1234)
        assert result == []

    @patch("vramtop.enrichment.model_files.is_same_user", return_value=True)
    def test_no_fd_dir(self, _mock_uid):
        with patch("os.listdir", side_effect=FileNotFoundError):
            result = scan_model_files(1234)
        assert result == []

    @pytest.mark.parametrize(
        "filename,expected_ext",
        [
            ("model.safetensors", ".safetensors"),
            ("model.gguf", ".gguf"),
            ("weights.pt", ".pt"),
            ("model.bin", ".bin"),
            ("model.onnx", ".onnx"),
            ("weights.pth", ".pth"),
            ("model.h5", ".h5"),
            ("model.tflite", ".tflite"),
        ],
    )
    @patch("vramtop.enrichment.model_files.is_same_user", return_value=True)
    def test_extension_matching(self, _mock_uid, filename, expected_ext):
        stat_result = MagicMock()
        stat_result.st_ino = 12345
        stat_result.st_size = 1024

        with (
            patch("os.listdir", return_value=["0"]),
            patch("os.readlink", return_value=f"/models/{filename}"),
            patch("os.stat", return_value=stat_result),
        ):
            result = scan_model_files(1234)

        assert len(result) == 1
        assert result[0].extension == expected_ext
        assert result[0].path == f"/models/{filename}"
        assert result[0].size_bytes == 1024

    @patch("vramtop.enrichment.model_files.is_same_user", return_value=True)
    def test_non_model_extension_skipped(self, _mock_uid):
        with (
            patch("os.listdir", return_value=["0"]),
            patch("os.readlink", return_value="/tmp/data.csv"),
        ):
            result = scan_model_files(1234)
        assert result == []

    @patch("vramtop.enrichment.model_files.is_same_user", return_value=True)
    def test_dedup_by_inode(self, _mock_uid):
        stat_result = MagicMock()
        stat_result.st_ino = 99999
        stat_result.st_size = 2048

        with (
            patch("os.listdir", return_value=["0", "1", "2"]),
            patch("os.readlink", return_value="/models/weights.pt"),
            patch("os.stat", return_value=stat_result),
        ):
            result = scan_model_files(1234)

        # Same inode -> only one entry
        assert len(result) == 1

    @patch("vramtop.enrichment.model_files.is_same_user", return_value=True)
    def test_max_fd_limit(self, _mock_uid):
        # Create more entries than _MAX_FDS
        entries = [str(i) for i in range(_MAX_FDS + 100)]

        call_count = 0

        def mock_readlink(path):
            nonlocal call_count
            call_count += 1
            return f"/models/model_{call_count}.safetensors"

        def mock_stat(path):
            m = MagicMock()
            m.st_ino = call_count  # Unique inodes
            m.st_size = 100
            return m

        with (
            patch("os.listdir", return_value=entries),
            patch("os.readlink", side_effect=mock_readlink),
            patch("os.stat", side_effect=mock_stat),
        ):
            result = scan_model_files(1234)

        assert len(result) <= _MAX_FDS

    @patch("vramtop.enrichment.model_files.is_same_user", return_value=True)
    def test_readlink_failure_skipped(self, _mock_uid):
        with (
            patch("os.listdir", return_value=["0", "1"]),
            patch("os.readlink", side_effect=OSError("broken link")),
        ):
            result = scan_model_files(1234)
        assert result == []
