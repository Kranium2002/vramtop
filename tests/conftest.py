"""Shared test fixtures for vramtop tests."""

from __future__ import annotations

import os
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from tests.fixtures.nvml_responses import (
    SINGLE_GPU_COMPUTE,
    SINGLE_GPU_MIXED,
    MULTI_GPU,
    GPU_LOST_SCENARIO,
    EMPTY_GPU,
    FakeDeviceHandle,
    FakeGPUScenario,
)


def _build_nvml_mock(scenarios: list[FakeGPUScenario], *, gpu_lost_index: int | None = None) -> MagicMock:
    """Build a mock pynvml module wired to the given GPU scenarios."""
    mock = MagicMock()

    # Constants
    mock.NVML_TEMPERATURE_GPU = 0
    mock.NVML_ERROR_GPU_IS_LOST = 9
    mock.NVML_ERROR_UNKNOWN = 999

    # NVMLError that can be raised and caught
    class FakeNVMLError(Exception):
        def __init__(self, value: int, msg: str = "") -> None:
            self.value = value
            super().__init__(msg or f"NVML Error {value}")

    mock.NVMLError = FakeNVMLError

    # Lifecycle
    mock.nvmlInit.return_value = None
    mock.nvmlShutdown.return_value = None

    # Driver/NVML version
    mock.nvmlSystemGetDriverVersion.return_value = "535.129.03"
    mock.nvmlSystemGetNVMLVersion.return_value = "12.535.129.03"

    # Device count
    mock.nvmlDeviceGetCount.return_value = len(scenarios)

    # Map index → scenario
    scenario_map: dict[int, FakeGPUScenario] = {s.handle.index: s for s in scenarios}

    def get_handle(index: int) -> FakeDeviceHandle:
        if gpu_lost_index is not None and index == gpu_lost_index:
            raise FakeNVMLError(mock.NVML_ERROR_GPU_IS_LOST, "GPU is lost")
        if index not in scenario_map:
            raise FakeNVMLError(mock.NVML_ERROR_UNKNOWN, f"No GPU at index {index}")
        return scenario_map[index].handle

    def get_name(handle: FakeDeviceHandle) -> str:
        return scenario_map[handle.index].name

    def get_uuid(handle: FakeDeviceHandle) -> str:
        return scenario_map[handle.index].uuid

    def get_memory_info(handle: FakeDeviceHandle) -> Any:
        return scenario_map[handle.index].memory

    def get_utilization(handle: FakeDeviceHandle) -> Any:
        return scenario_map[handle.index].utilization

    def get_temperature(handle: FakeDeviceHandle, _sensor: int) -> int:
        return scenario_map[handle.index].temperature

    def get_power(handle: FakeDeviceHandle) -> int:
        return scenario_map[handle.index].power_mw

    def get_compute_procs(handle: FakeDeviceHandle) -> list[Any]:
        return list(scenario_map[handle.index].compute_processes)

    def get_graphics_procs(handle: FakeDeviceHandle) -> list[Any]:
        return list(scenario_map[handle.index].graphics_processes)

    mock.nvmlDeviceGetHandleByIndex.side_effect = get_handle
    mock.nvmlDeviceGetName.side_effect = get_name
    mock.nvmlDeviceGetUUID.side_effect = get_uuid
    mock.nvmlDeviceGetMemoryInfo.side_effect = get_memory_info
    mock.nvmlDeviceGetMemoryInfo_v2 = None  # Not available by default, triggers v1 fallback
    mock.nvmlDeviceGetUtilizationRates.side_effect = get_utilization
    mock.nvmlDeviceGetTemperature.side_effect = get_temperature
    mock.nvmlDeviceGetPowerUsage.side_effect = get_power
    mock.nvmlDeviceGetComputeRunningProcesses.side_effect = get_compute_procs
    mock.nvmlDeviceGetGraphicsRunningProcesses.side_effect = get_graphics_procs

    return mock


def _build_nvml_mock_with_v2(scenarios: list[FakeGPUScenario]) -> MagicMock:
    """Build a mock with v2 memory info available."""
    mock = _build_nvml_mock(scenarios)
    scenario_map = {s.handle.index: s for s in scenarios}

    def get_memory_info_v2(handle: FakeDeviceHandle) -> Any:
        return scenario_map[handle.index].memory

    mock.nvmlDeviceGetMemoryInfo_v2 = MagicMock(side_effect=get_memory_info_v2)
    return mock


@pytest.fixture()
def mock_nvml_single_gpu():
    """Patch pynvml with a single compute-only GPU."""
    mock = _build_nvml_mock([SINGLE_GPU_COMPUTE])
    with patch.dict("sys.modules", {"pynvml": mock}):
        yield mock


@pytest.fixture()
def mock_nvml_mixed_gpu():
    """Patch pynvml with a single GPU that has both compute and graphics processes."""
    mock = _build_nvml_mock([SINGLE_GPU_MIXED])
    with patch.dict("sys.modules", {"pynvml": mock}):
        yield mock


@pytest.fixture()
def mock_nvml_multi_gpu():
    """Patch pynvml with two GPUs."""
    mock = _build_nvml_mock(MULTI_GPU)
    with patch.dict("sys.modules", {"pynvml": mock}):
        yield mock


@pytest.fixture()
def mock_nvml_gpu_lost():
    """Patch pynvml where GPU 0 raises GPU_IS_LOST."""
    mock = _build_nvml_mock([GPU_LOST_SCENARIO], gpu_lost_index=0)
    with patch.dict("sys.modules", {"pynvml": mock}):
        yield mock


@pytest.fixture()
def mock_nvml_empty_gpu():
    """Patch pynvml with a single GPU with no processes."""
    mock = _build_nvml_mock([EMPTY_GPU])
    with patch.dict("sys.modules", {"pynvml": mock}):
        yield mock


@pytest.fixture()
def mock_nvml_v2_memory():
    """Patch pynvml with v2 memory info available."""
    mock = _build_nvml_mock_with_v2([SINGLE_GPU_COMPUTE])
    with patch.dict("sys.modules", {"pynvml": mock}):
        yield mock


@pytest.fixture()
def mock_proc_stat(tmp_path):
    """Create fake /proc/<pid>/stat files for process identity testing.

    Returns a helper function to create stat files for given PIDs.
    """
    proc_dir = tmp_path / "proc"
    proc_dir.mkdir()

    def create_stat(pid: int, starttime: int = 12345) -> None:
        pid_dir = proc_dir / str(pid)
        pid_dir.mkdir(exist_ok=True)
        # Format: pid (comm) S ... field22=starttime
        # Fields after comm: state, ppid, pgrp, session, tty_nr, tpgid,
        #   flags, minflt, cminflt, majflt, cmajflt, utime, stime, cutime,
        #   cstime, priority, nice, num_threads, itrealvalue, starttime
        # That's 20 fields (indices 0-19), starttime is at index 19
        fields_after_comm = ["S"] + ["0"] * 18 + [str(starttime)]
        stat_content = f"{pid} (python) {' '.join(fields_after_comm)}\n"
        (pid_dir / "stat").write_text(stat_content)
        # Also create a cmdline file
        (pid_dir / "cmdline").write_bytes(b"python\x00train.py\x00--epochs=10")

    def create_cmdline(pid: int, cmdline_bytes: bytes) -> None:
        pid_dir = proc_dir / str(pid)
        pid_dir.mkdir(exist_ok=True)
        (pid_dir / "cmdline").write_bytes(cmdline_bytes)

    return type("ProcHelper", (), {
        "proc_dir": proc_dir,
        "create_stat": staticmethod(create_stat),
        "create_cmdline": staticmethod(create_cmdline),
    })
