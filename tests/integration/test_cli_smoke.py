"""End-to-end CLI smoke tests for vramtop.

These tests require a real NVIDIA GPU and are marked with @pytest.mark.gpu.
"""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.gpu
class TestCLIHelp:
    """Test --help output."""

    def test_help_exit_code(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "vramtop", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0

    def test_help_contains_expected_flags(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "vramtop", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        output = result.stdout
        assert "--no-kill" in output
        assert "--config" in output
        assert "--export-csv" in output
        assert "--accessible" in output


@pytest.mark.gpu
class TestCLIVersion:
    """Test --version output."""

    def test_version_exit_code(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "vramtop", "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0

    def test_version_contains_version_string(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "vramtop", "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        output = result.stdout
        assert "vramtop" in output
        # Should contain a version like X.Y.Z
        from vramtop import __version__

        assert __version__ in output


@pytest.mark.gpu
class TestCSVExportWithRealGPU:
    """Test CSV export using real NVML backend data."""

    def test_csv_roundtrip_with_nvml_snapshot(self, tmp_path: Path) -> None:
        """Take a real NVML snapshot and write it to CSV, then verify."""
        from vramtop.backends.nvidia import NVMLClient
        from vramtop.export.csv_logger import CSVLogger

        backend = NVMLClient()
        backend.initialize()
        try:
            snapshot = backend.snapshot()
        finally:
            backend.shutdown()

        # Verify snapshot has at least one device
        assert len(snapshot.devices) >= 1

        csv_file = tmp_path / "gpu_export.csv"
        logger = CSVLogger(csv_file)
        logger.start()
        logger.write_snapshot(snapshot)
        logger.stop()

        # Read back and verify
        assert csv_file.exists()
        rows = list(csv.DictReader(csv_file.open()))
        assert len(rows) >= 1

        first_row = rows[0]
        assert first_row["gpu_index"] == "0"
        assert first_row["gpu_name"] != ""
        assert int(first_row["gpu_total_bytes"]) > 0
        assert int(first_row["gpu_used_bytes"]) >= 0
        assert float(first_row["wall_time"]) > 0


@pytest.mark.gpu
class TestExportManagerIntegration:
    """Test ExportManager with real NVML data."""

    def test_export_manager_csv_lifecycle(self, tmp_path: Path) -> None:
        """Create an ExportManager, feed it a real snapshot, verify CSV."""
        from vramtop.backends.nvidia import NVMLClient
        from vramtop.config import ExportConfig
        from vramtop.export import ExportManager

        backend = NVMLClient()
        backend.initialize()
        try:
            snapshot = backend.snapshot()
        finally:
            backend.shutdown()

        csv_file = tmp_path / "manager_export.csv"
        config = ExportConfig()
        manager = ExportManager(config, csv_path=csv_file)
        manager.start()
        manager.update_snapshot(snapshot)
        manager.stop()

        # Verify CSV file
        assert csv_file.exists()
        with csv_file.open() as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames
            assert headers is not None
            assert "wall_time" in headers
            assert "gpu_index" in headers
            assert "gpu_name" in headers
            assert "gpu_used_bytes" in headers
            assert "gpu_total_bytes" in headers
            assert "gpu_util_percent" in headers
            assert "pid" in headers
            assert "process_name" in headers
            assert "process_vram_bytes" in headers
            assert "process_type" in headers

            rows = list(reader)
            assert len(rows) >= 1

            first_row = rows[0]
            # GPU data should be populated
            assert first_row["gpu_index"] == "0"
            assert first_row["gpu_name"] != ""
            assert int(first_row["gpu_total_bytes"]) > 0
            assert float(first_row["wall_time"]) > 0
