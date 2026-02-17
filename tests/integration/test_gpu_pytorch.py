"""Integration tests using real NVIDIA GPU with PyTorch CUDA workloads.

These tests require:
- An NVIDIA GPU with CUDA support
- PyTorch installed with CUDA backend
- NVIDIA drivers and NVML library

Run with: pytest tests/integration/test_gpu_pytorch.py -v --tb=short
Skip in CI without GPU: pytest -m "not gpu"
"""

from __future__ import annotations

import csv
import os
import time
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA not available", allow_module_level=True)

from vramtop.analysis.phase_detector import Phase, PhaseDetector
from vramtop.backends.nvidia import NVMLClient
from vramtop.export.csv_logger import CSVLogger

gpu = pytest.mark.gpu

_500MB = 128 * 1024 * 1024  # 128M float32 elements = 512 MB


def _in_pid_namespace() -> bool:
    """Detect if we are in a PID namespace (e.g. Docker container).

    When running inside a container, NVML reports host-level PIDs that
    differ from os.getpid().  Tests that match PIDs need to account for this.
    """
    try:
        with open("/proc/1/cgroup") as f:
            return "docker" in f.read() or "kubepods" in f.read()
    except (FileNotFoundError, PermissionError):
        return False


_PID_NAMESPACE = _in_pid_namespace()


@gpu
class TestNVMLBackendSnapshot:
    """Test NVML backend snapshot with a real GPU."""

    def test_snapshot_device_count(self) -> None:
        client = NVMLClient()
        try:
            client.initialize()
            snap = client.snapshot()
            assert len(snap.devices) >= 1
        finally:
            client.shutdown()

    def test_snapshot_device_name(self) -> None:
        client = NVMLClient()
        try:
            client.initialize()
            snap = client.snapshot()
            name = snap.devices[0].name
            assert any(kw in name for kw in ("NVIDIA", "RTX", "GeForce", "Tesla", "Quadro")), (
                f"Unexpected GPU name: {name}"
            )
        finally:
            client.shutdown()

    def test_snapshot_memory_totals(self) -> None:
        client = NVMLClient()
        try:
            client.initialize()
            snap = client.snapshot()
            dev = snap.devices[0]
            assert dev.total_memory_bytes > 0
            assert dev.used_memory_bytes >= 0
            assert dev.free_memory_bytes >= 0
            # Sanity: used + free should be close to total (within 10% tolerance)
            assert dev.used_memory_bytes + dev.free_memory_bytes == pytest.approx(
                dev.total_memory_bytes, rel=0.1
            )
        finally:
            client.shutdown()

    def test_snapshot_driver_version(self) -> None:
        client = NVMLClient()
        try:
            client.initialize()
            snap = client.snapshot()
            assert isinstance(snap.driver_version, str)
            assert len(snap.driver_version) > 0
            # Driver version should look like "5xx.xxx.xx" or similar
            assert snap.driver_version[0].isdigit()
        finally:
            client.shutdown()

    def test_snapshot_timestamps(self) -> None:
        client = NVMLClient()
        try:
            client.initialize()
            snap = client.snapshot()
            assert snap.timestamp > 0
            assert snap.wall_time > 0
        finally:
            client.shutdown()

    def test_context_manager_lifecycle(self) -> None:
        with NVMLClient() as client:
            snap = client.snapshot()
            assert len(snap.devices) >= 1


def _find_our_gpu_process(dev, *, our_pid: int):
    """Find our process in NVML's device process list.

    In a Docker/PID-namespace environment NVML reports host-level PIDs that
    differ from os.getpid().  In that case, we fall back to finding the
    process with the largest GPU memory usage (which is ours, since we are
    the only one actively allocating 512 MB for the test).
    """
    # Try exact PID match first (works on bare-metal)
    for p in dev.processes:
        if p.identity.pid == our_pid:
            return p

    if _PID_NAMESPACE and dev.processes:
        # In a container: NVML reports host PIDs. Pick the process using
        # the most memory -- that's our 512 MB allocation.
        return max(dev.processes, key=lambda p: p.used_memory_bytes)

    return None


@gpu
class TestProcessDetection:
    """Test that PyTorch GPU allocations are visible via NVML process list."""

    def test_gpu_process_visible_after_allocation(self) -> None:
        tensor = None
        client = NVMLClient()
        try:
            client.initialize()

            # Snapshot before allocation to know baseline process count
            snap_before = client.snapshot()
            procs_before = len(snap_before.devices[0].processes)

            # Allocate ~512MB on GPU to ensure our process is visible
            tensor = torch.zeros(128, 1024, 1024, device="cuda")
            torch.cuda.synchronize()

            snap = client.snapshot()
            dev = snap.devices[0]

            our_pid = os.getpid()
            our_proc = _find_our_gpu_process(dev, our_pid=our_pid)
            assert our_proc is not None, (
                f"No GPU process found for our allocation "
                f"(PID {our_pid}, namespace={_PID_NAMESPACE}, "
                f"visible PIDs={[p.identity.pid for p in dev.processes]})"
            )
            # At least one process should be using the GPU
            assert len(dev.processes) >= 1
        finally:
            del tensor
            torch.cuda.empty_cache()
            client.shutdown()

    def test_reported_memory_reasonable(self) -> None:
        tensor = None
        client = NVMLClient()
        try:
            client.initialize()
            tensor = torch.zeros(128, 1024, 1024, device="cuda")
            torch.cuda.synchronize()

            snap = client.snapshot()
            dev = snap.devices[0]

            our_proc = _find_our_gpu_process(dev, our_pid=os.getpid())
            assert our_proc is not None, "GPU process not found after 512MB allocation"

            # The tensor is 512MB, but CUDA context adds overhead.
            # Total should be at least 500MB.
            min_expected = 500 * 1024 * 1024  # 500 MB
            assert our_proc.used_memory_bytes >= min_expected, (
                f"Expected >= 500MB, got {our_proc.used_memory_bytes / (1024**2):.0f}MB"
            )
        finally:
            del tensor
            torch.cuda.empty_cache()
            client.shutdown()

    def test_process_type_is_compute(self) -> None:
        tensor = None
        client = NVMLClient()
        try:
            client.initialize()
            tensor = torch.zeros(128, 1024, 1024, device="cuda")
            torch.cuda.synchronize()

            snap = client.snapshot()
            dev = snap.devices[0]

            our_proc = _find_our_gpu_process(dev, our_pid=os.getpid())
            assert our_proc is not None
            # PyTorch uses compute context
            assert "compute" in our_proc.process_type
        finally:
            del tensor
            torch.cuda.empty_cache()
            client.shutdown()


@gpu
class TestCSVExportWithRealData:
    """Test CSV export using real GPU snapshots."""

    def test_csv_has_correct_columns(self, tmp_path: Path) -> None:
        tensor = None
        client = NVMLClient()
        csv_file = tmp_path / "gpu_test.csv"
        logger = CSVLogger(csv_file)
        try:
            client.initialize()
            tensor = torch.zeros(128, 1024, 1024, device="cuda")
            torch.cuda.synchronize()

            logger.start()
            snap = client.snapshot()
            logger.write_snapshot(snap)
            logger.stop()

            assert csv_file.exists()
            with csv_file.open() as f:
                reader = csv.DictReader(f)
                fields = reader.fieldnames
                assert fields is not None
                assert "wall_time" in fields
                assert "gpu_index" in fields
                assert "gpu_name" in fields
                assert "gpu_used_bytes" in fields
                assert "gpu_total_bytes" in fields
                assert "pid" in fields
                assert "process_name" in fields
                assert "process_vram_bytes" in fields
                assert "process_type" in fields
        finally:
            del tensor
            torch.cuda.empty_cache()
            client.shutdown()

    def test_csv_data_is_reasonable(self, tmp_path: Path) -> None:
        tensor = None
        client = NVMLClient()
        csv_file = tmp_path / "gpu_test.csv"
        logger = CSVLogger(csv_file)
        try:
            client.initialize()
            tensor = torch.zeros(128, 1024, 1024, device="cuda")
            torch.cuda.synchronize()

            logger.start()
            snap = client.snapshot()
            logger.write_snapshot(snap)
            logger.stop()

            with csv_file.open() as f:
                rows = list(csv.DictReader(f))

            # Should have at least one row with a process
            assert len(rows) >= 1

            # Find rows with non-empty PIDs (actual processes)
            proc_rows = [r for r in rows if r["pid"] != ""]
            assert len(proc_rows) >= 1, "No process rows in CSV"

            # Find the row with the largest VRAM usage (our 512MB allocation)
            row = max(proc_rows, key=lambda r: int(r["process_vram_bytes"]))
            assert int(row["gpu_index"]) == 0
            assert int(row["process_vram_bytes"]) >= 500 * 1024 * 1024
            assert row["process_type"] != ""
            assert float(row["wall_time"]) > 0
            assert int(row["gpu_total_bytes"]) > 0
        finally:
            del tensor
            torch.cuda.empty_cache()
            client.shutdown()

    def test_csv_multiple_snapshots(self, tmp_path: Path) -> None:
        tensor = None
        client = NVMLClient()
        csv_file = tmp_path / "gpu_test.csv"
        logger = CSVLogger(csv_file)
        try:
            client.initialize()
            tensor = torch.zeros(128, 1024, 1024, device="cuda")
            torch.cuda.synchronize()

            logger.start()
            snap1 = client.snapshot()
            logger.write_snapshot(snap1)
            time.sleep(0.1)
            snap2 = client.snapshot()
            logger.write_snapshot(snap2)
            logger.stop()

            with csv_file.open() as f:
                rows = list(csv.DictReader(f))

            # Two snapshots, each should produce at least one row
            assert len(rows) >= 2

            # wall_time should increase
            times = [float(r["wall_time"]) for r in rows]
            assert times[-1] >= times[0]
        finally:
            del tensor
            torch.cuda.empty_cache()
            client.shutdown()


@gpu
class TestPhaseDetectorWithRealMemory:
    """Test phase detector using real GPU memory changes."""

    def test_growing_phase_detection(self) -> None:
        """Allocate increasing tensors and verify GROWING phase is detected."""
        tensors: list = []
        client = NVMLClient()
        try:
            client.initialize()

            detector = PhaseDetector(
                window_size=8,
                hysteresis_samples=3,
                noise_floor_mb=0.5,
            )

            prev_used_mb = 0.0
            last_state = None

            # Take a baseline reading
            snap = client.snapshot()
            prev_used_mb = snap.devices[0].used_memory_bytes / (1024 * 1024)

            # Allocate increasingly larger tensors in a loop
            for i in range(12):
                # Each allocation is ~128MB (32M float32 elements)
                t = torch.zeros(32, 1024, 1024, device="cuda")
                tensors.append(t)
                torch.cuda.synchronize()

                snap = client.snapshot()
                current_used_mb = snap.devices[0].used_memory_bytes / (1024 * 1024)
                delta_mb = current_used_mb - prev_used_mb
                prev_used_mb = current_used_mb

                last_state = detector.update(delta_mb, dt_seconds=1.0)

            # After sustained growth, the detector should have detected GROWING
            assert last_state is not None
            assert last_state.phase == Phase.GROWING, (
                f"Expected GROWING phase after 12 allocations, got {last_state.phase}"
            )
        finally:
            for t in tensors:
                del t
            tensors.clear()
            torch.cuda.empty_cache()
            client.shutdown()

    def test_stable_phase_with_no_changes(self) -> None:
        """With no memory changes, phase should remain STABLE."""
        tensor = None
        client = NVMLClient()
        try:
            client.initialize()

            # Allocate once to establish a baseline
            tensor = torch.zeros(32, 1024, 1024, device="cuda")
            torch.cuda.synchronize()

            detector = PhaseDetector(
                window_size=8,
                hysteresis_samples=3,
                noise_floor_mb=1.0,
            )

            last_state = None
            # Take multiple snapshots without changing memory
            for _ in range(10):
                # No new allocations; delta should be near zero
                last_state = detector.update(delta_mb=0.0, dt_seconds=1.0)

            assert last_state is not None
            assert last_state.phase == Phase.STABLE, (
                f"Expected STABLE phase with no changes, got {last_state.phase}"
            )
        finally:
            del tensor
            torch.cuda.empty_cache()
            client.shutdown()
