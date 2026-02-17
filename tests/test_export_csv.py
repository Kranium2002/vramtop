"""Tests for CSV export logger."""

from __future__ import annotations

import csv
from pathlib import Path

from vramtop.backends.base import (
    GPUDevice,
    GPUProcess,
    MemorySnapshot,
    ProcessIdentity,
)
from vramtop.export.csv_logger import CSVLogger


def _make_snapshot(
    *,
    wall_time: float = 1700000000.0,
    gpu_name: str = "Test GPU",
    used: int = 4_000_000_000,
    total: int = 8_000_000_000,
    processes: tuple[GPUProcess, ...] | None = None,
) -> MemorySnapshot:
    if processes is None:
        processes = (
            GPUProcess(
                identity=ProcessIdentity(pid=100, starttime=999),
                name="python",
                used_memory_bytes=2_000_000_000,
                process_type="compute",
            ),
        )
    device = GPUDevice(
        index=0,
        uuid="GPU-TEST-0",
        name=gpu_name,
        total_memory_bytes=total,
        used_memory_bytes=used,
        free_memory_bytes=total - used,
        gpu_util_percent=75,
        mem_util_percent=50,
        temperature_celsius=65,
        power_watts=200.0,
        processes=processes,
    )
    return MemorySnapshot(
        timestamp=1000.0,
        wall_time=wall_time,
        devices=(device,),
        driver_version="535.0",
        nvml_version="12.0",
    )


class TestCSVLogger:
    def test_creates_file_with_header(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "test.csv"
        logger = CSVLogger(csv_file)
        logger.start()
        logger.stop()

        rows = list(csv.DictReader(csv_file.open()))
        assert len(rows) == 0  # No data rows yet, just header
        header = csv_file.read_text().strip()
        assert "wall_time" in header
        assert "gpu_index" in header
        assert "pid" in header

    def test_write_snapshot_with_processes(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "test.csv"
        logger = CSVLogger(csv_file)
        logger.start()

        snap = _make_snapshot()
        logger.write_snapshot(snap)
        logger.stop()

        rows = list(csv.DictReader(csv_file.open()))
        assert len(rows) == 1
        row = rows[0]
        assert row["gpu_index"] == "0"
        assert row["gpu_name"] == "Test GPU"
        assert row["pid"] == "100"
        assert row["process_name"] == "python"
        assert row["process_vram_bytes"] == "2000000000"
        assert row["process_type"] == "compute"

    def test_write_snapshot_no_processes(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "test.csv"
        logger = CSVLogger(csv_file)
        logger.start()

        snap = _make_snapshot(processes=())
        logger.write_snapshot(snap)
        logger.stop()

        rows = list(csv.DictReader(csv_file.open()))
        assert len(rows) == 1
        row = rows[0]
        assert row["gpu_index"] == "0"
        assert row["pid"] == ""
        assert row["process_name"] == ""

    def test_multiple_snapshots_append(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "test.csv"
        logger = CSVLogger(csv_file)
        logger.start()

        logger.write_snapshot(_make_snapshot(wall_time=1.0))
        logger.write_snapshot(_make_snapshot(wall_time=2.0))
        logger.stop()

        rows = list(csv.DictReader(csv_file.open()))
        assert len(rows) == 2
        assert rows[0]["wall_time"] == "1.0"
        assert rows[1]["wall_time"] == "2.0"

    def test_multiple_processes_per_gpu(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "test.csv"
        logger = CSVLogger(csv_file)
        logger.start()

        procs = (
            GPUProcess(
                identity=ProcessIdentity(pid=100, starttime=999),
                name="train.py",
                used_memory_bytes=3_000_000_000,
                process_type="compute",
            ),
            GPUProcess(
                identity=ProcessIdentity(pid=200, starttime=888),
                name="eval.py",
                used_memory_bytes=1_000_000_000,
                process_type="compute",
            ),
        )
        snap = _make_snapshot(processes=procs)
        logger.write_snapshot(snap)
        logger.stop()

        rows = list(csv.DictReader(csv_file.open()))
        assert len(rows) == 2
        assert rows[0]["pid"] == "100"
        assert rows[1]["pid"] == "200"

    def test_write_before_start_is_noop(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "test.csv"
        logger = CSVLogger(csv_file)
        # Don't call start()
        logger.write_snapshot(_make_snapshot())
        # File shouldn't exist
        assert not csv_file.exists()

    def test_stop_is_idempotent(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "test.csv"
        logger = CSVLogger(csv_file)
        logger.start()
        logger.stop()
        logger.stop()  # Should not raise

    def test_start_is_idempotent(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "test.csv"
        logger = CSVLogger(csv_file)
        logger.start()
        logger.start()  # Should not raise or create duplicate headers
        logger.write_snapshot(_make_snapshot())
        logger.stop()

        text = csv_file.read_text()
        # Header should appear only once
        assert text.count("wall_time") == 1
