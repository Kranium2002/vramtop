"""Tests for the NVIDIA NVML backend."""

from __future__ import annotations

import importlib
import sys
from unittest.mock import patch, MagicMock

import pytest


def _fresh_nvidia_module(mock_pynvml: MagicMock):
    """Import a fresh nvidia module with the mocked pynvml in sys.modules."""
    # Ensure pynvml mock is in sys.modules
    sys.modules["pynvml"] = mock_pynvml
    # Force re-import of nvidia module to pick up the mock
    mod_name = "vramtop.backends.nvidia"
    if mod_name in sys.modules:
        del sys.modules[mod_name]
    mod = importlib.import_module(mod_name)
    return mod


class TestDeviceCount:
    """Test device_count() returns correct number of GPUs."""

    def test_single_gpu(self, mock_nvml_single_gpu):
        mod = _fresh_nvidia_module(mock_nvml_single_gpu)
        client = mod.NVMLClient()
        client.initialize()
        assert client.device_count() == 1
        client.shutdown()

    def test_multi_gpu(self, mock_nvml_multi_gpu):
        mod = _fresh_nvidia_module(mock_nvml_multi_gpu)
        client = mod.NVMLClient()
        client.initialize()
        assert client.device_count() == 2
        client.shutdown()

    def test_context_manager(self, mock_nvml_single_gpu):
        mod = _fresh_nvidia_module(mock_nvml_single_gpu)
        with mod.NVMLClient() as client:
            assert client.device_count() == 1


class TestSnapshotFields:
    """Test that snapshot returns correct fields."""

    def test_single_gpu_snapshot_structure(self, mock_nvml_single_gpu):
        mod = _fresh_nvidia_module(mock_nvml_single_gpu)
        with mod.NVMLClient() as client:
            snap = client.snapshot()

        assert snap.driver_version == "535.129.03"
        assert snap.nvml_version == "12.535.129.03"
        assert len(snap.devices) == 1
        assert snap.timestamp > 0
        assert snap.wall_time > 0

    def test_device_fields(self, mock_nvml_single_gpu):
        mod = _fresh_nvidia_module(mock_nvml_single_gpu)
        with mod.NVMLClient() as client:
            snap = client.snapshot()

        dev = snap.devices[0]
        assert dev.index == 0
        assert dev.name == "NVIDIA A100-SXM4-80GB"
        assert dev.uuid == "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
        assert dev.total_memory_bytes == 85899345920
        # used = sum of process allocations (v1 fallback excludes
        # driver-reserved memory).  free = NVML's free (truly allocatable).
        assert dev.used_memory_bytes == 30_000_000_000
        assert dev.free_memory_bytes == 42949672960  # mem_info.free
        assert dev.gpu_util_percent == 75
        assert dev.mem_util_percent == 60
        assert dev.temperature_celsius == 65
        assert dev.power_watts == pytest.approx(250.0)

    def test_multi_gpu_snapshot(self, mock_nvml_multi_gpu):
        mod = _fresh_nvidia_module(mock_nvml_multi_gpu)
        with mod.NVMLClient() as client:
            snap = client.snapshot()

        assert len(snap.devices) == 2
        assert snap.devices[0].index == 0
        assert snap.devices[1].index == 1
        assert snap.devices[0].gpu_util_percent == 90
        assert snap.devices[1].gpu_util_percent == 20

    def test_empty_gpu_no_processes(self, mock_nvml_empty_gpu):
        mod = _fresh_nvidia_module(mock_nvml_empty_gpu)
        with mod.NVMLClient() as client:
            snap = client.snapshot()

        dev = snap.devices[0]
        assert dev.name == "NVIDIA T4"
        assert len(dev.processes) == 0


class TestComputeGraphicsMerge:
    """Test compute + graphics process merge/dedup — critical functionality."""

    def test_compute_only_processes(self, mock_nvml_single_gpu):
        """Processes only in compute list get type='compute'."""
        mod = _fresh_nvidia_module(mock_nvml_single_gpu)
        with mod.NVMLClient() as client:
            snap = client.snapshot()

        procs = snap.devices[0].processes
        assert len(procs) == 2
        for proc in procs:
            assert proc.process_type == "compute"

    def test_mixed_process_merge(self, mock_nvml_mixed_gpu):
        """PID 2001 in both lists → compute+graphics with summed memory."""
        mod = _fresh_nvidia_module(mock_nvml_mixed_gpu)
        with mod.NVMLClient() as client:
            snap = client.snapshot()

        procs = snap.devices[0].processes
        procs_by_pid = {p.identity.pid: p for p in procs}

        # PID 2001 is in both compute and graphics lists
        assert 2001 in procs_by_pid
        merged = procs_by_pid[2001]
        assert merged.process_type == "compute+graphics"
        # Memory should be summed: 4GB (compute) + 1GB (graphics) = 5GB
        assert merged.used_memory_bytes == 5_000_000_000

        # PID 2002 is only in graphics list
        assert 2002 in procs_by_pid
        gfx_only = procs_by_pid[2002]
        assert gfx_only.process_type == "graphics"
        assert gfx_only.used_memory_bytes == 2_000_000_000

    def test_dedup_count(self, mock_nvml_mixed_gpu):
        """3 process entries (2 compute, 1 graphics with overlap) → 2 unique processes."""
        mod = _fresh_nvidia_module(mock_nvml_mixed_gpu)
        with mod.NVMLClient() as client:
            snap = client.snapshot()

        # 2001 (compute + graphics merged) and 2002 (graphics only)
        assert len(snap.devices[0].processes) == 2


class TestGPULostHandling:
    """Test GPU-lost error translation."""

    def test_gpu_lost_raises_gpu_lost_error(self, mock_nvml_gpu_lost):
        mod = _fresh_nvidia_module(mock_nvml_gpu_lost)
        from vramtop.backends.base import GPULostError

        with mod.NVMLClient() as client:
            with pytest.raises(GPULostError):
                client.snapshot()


class TestDriverErrorRetry:
    """Test NVML_ERROR_UNKNOWN retry with exponential backoff."""

    def test_unknown_error_retries_then_raises(self, mock_nvml_single_gpu):
        mod = _fresh_nvidia_module(mock_nvml_single_gpu)
        pynvml_mock = sys.modules["pynvml"]
        from vramtop.backends.base import DriverError

        # Make device count always fail with UNKNOWN
        err = pynvml_mock.NVMLError(pynvml_mock.NVML_ERROR_UNKNOWN, "Unknown error")
        pynvml_mock.nvmlDeviceGetCount.side_effect = err

        with mod.NVMLClient() as client:
            with patch("time.sleep"):  # Don't actually sleep
                with pytest.raises(DriverError):
                    client.device_count()

        # Verify it was called 3 times (retries)
        assert pynvml_mock.nvmlDeviceGetCount.call_count == 3

    def test_unknown_error_succeeds_on_retry(self, mock_nvml_single_gpu):
        mod = _fresh_nvidia_module(mock_nvml_single_gpu)
        pynvml_mock = sys.modules["pynvml"]

        err = pynvml_mock.NVMLError(pynvml_mock.NVML_ERROR_UNKNOWN, "Unknown error")
        # Fail first, succeed on second attempt
        pynvml_mock.nvmlDeviceGetCount.side_effect = [err, 1]

        with mod.NVMLClient() as client:
            with patch("time.sleep"):
                count = client.device_count()

        assert count == 1


class TestProcessVanish:
    """Test handling of processes that disappear mid-snapshot."""

    def test_process_with_vanished_proc(self, mock_nvml_single_gpu):
        """Process identity should fall back to starttime=0 for vanished processes."""
        mod = _fresh_nvidia_module(mock_nvml_single_gpu)

        # Patch get_process_identity to return None (process vanished)
        with patch("vramtop.backends.nvidia._get_identity") as mock_ident:
            from vramtop.backends.base import ProcessIdentity
            mock_ident.return_value = ProcessIdentity(pid=0, starttime=0)

            with mod.NVMLClient() as client:
                snap = client.snapshot()

            procs = snap.devices[0].processes
            assert len(procs) == 2

    def test_process_name_fallback(self, mock_nvml_single_gpu):
        """Process name should fall back to 'PID <n>' when /proc is inaccessible."""
        mod = _fresh_nvidia_module(mock_nvml_single_gpu)

        with patch("vramtop.backends.nvidia._resolve_process_name") as mock_name:
            mock_name.side_effect = lambda pid: f"PID {pid}"

            with mod.NVMLClient() as client:
                snap = client.snapshot()

            for proc in snap.devices[0].processes:
                assert proc.name.startswith("PID ")


class TestMemoryInfoV2Fallback:
    """Test v2 → v1 memory info fallback."""

    def test_v2_used_when_available(self, mock_nvml_v2_memory):
        mod = _fresh_nvidia_module(mock_nvml_v2_memory)
        with mod.NVMLClient() as client:
            snap = client.snapshot()

        dev = snap.devices[0]
        assert dev.total_memory_bytes == 85899345920

    def test_v1_fallback_when_v2_missing(self, mock_nvml_single_gpu):
        """When nvmlDeviceGetMemoryInfo_v2 is None, v1 is used."""
        mod = _fresh_nvidia_module(mock_nvml_single_gpu)
        pynvml_mock = sys.modules["pynvml"]
        # Ensure v2 is not available
        assert pynvml_mock.nvmlDeviceGetMemoryInfo_v2 is None

        with mod.NVMLClient() as client:
            snap = client.snapshot()

        dev = snap.devices[0]
        assert dev.total_memory_bytes == 85899345920


class TestLifecycle:
    """Test init/shutdown lifecycle."""

    def test_double_initialize_is_idempotent(self, mock_nvml_single_gpu):
        mod = _fresh_nvidia_module(mock_nvml_single_gpu)
        client = mod.NVMLClient()
        client.initialize()
        client.initialize()  # should not raise
        client.shutdown()

    def test_shutdown_without_init(self, mock_nvml_single_gpu):
        mod = _fresh_nvidia_module(mock_nvml_single_gpu)
        client = mod.NVMLClient()
        client.shutdown()  # should not raise

    def test_init_failure_raises_backend_error(self, mock_nvml_single_gpu):
        mod = _fresh_nvidia_module(mock_nvml_single_gpu)
        pynvml_mock = sys.modules["pynvml"]
        from vramtop.backends.base import BackendError

        pynvml_mock.nvmlInit.side_effect = pynvml_mock.NVMLError(1, "No driver")

        client = mod.NVMLClient()
        with pytest.raises(BackendError, match="Failed to initialize NVML"):
            client.initialize()


class TestGetBackendFactory:
    """Test the get_backend() factory function."""

    def test_get_backend_returns_nvml_client(self, mock_nvml_single_gpu):
        _fresh_nvidia_module(mock_nvml_single_gpu)
        from vramtop.backends import get_backend

        backend = get_backend()
        assert type(backend).__name__ == "NVMLClient"
