"""NVIDIA GPU backend using nvidia-ml-py (pynvml)."""

from __future__ import annotations

import contextlib
import logging
import time
from typing import Any

import pynvml  # type: ignore[import-untyped]

from vramtop.backends.base import (
    BackendError,
    DriverError,
    GPUBackend,
    GPUDevice,
    GPULostError,
    GPUProcess,
    MemorySnapshot,
    ProcessIdentity,
)

logger = logging.getLogger(__name__)

# Maximum retries for NVML_ERROR_UNKNOWN before raising DriverError.
_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 0.1  # seconds


def _resolve_process_name(pid: int) -> str:
    """Resolve a human-readable process name for a PID.

    Strategy:
    1. If same user, read /proc/<pid>/cmdline and sanitize.
    2. Otherwise, return "PID <pid>".
    """
    try:
        from vramtop.permissions import is_same_user
        from vramtop.sanitize import sanitize_process_name
    except ImportError:
        return f"PID {pid}"

    if not is_same_user(pid):
        return f"PID {pid}"

    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            raw = f.read(4096)
        if not raw:
            return f"PID {pid}"
        # cmdline is null-separated; join with spaces and sanitize
        cmdline = raw.replace(b"\x00", b" ").decode("utf-8", errors="replace").strip()
        name = sanitize_process_name(cmdline)
        return name if name else f"PID {pid}"
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return f"PID {pid}"


def _get_identity(pid: int) -> ProcessIdentity:
    """Get (PID, starttime) identity, falling back to starttime=0."""
    try:
        from vramtop.process_identity import get_process_identity
    except ImportError:
        return ProcessIdentity(pid=pid, starttime=0)

    ident = get_process_identity(pid)
    if ident is None:
        return ProcessIdentity(pid=pid, starttime=0)
    return ident


def _translate_nvml_error(err: pynvml.NVMLError) -> BackendError:
    """Translate an NVML error to our exception hierarchy."""
    code = err.value
    if code == pynvml.NVML_ERROR_GPU_IS_LOST:
        return GPULostError(str(err))
    return DriverError(str(err))


def _call_with_retry(func: Any, *args: Any) -> Any:
    """Call an NVML function, retrying on NVML_ERROR_UNKNOWN with exponential backoff."""
    last_err: pynvml.NVMLError | None = None
    for attempt in range(_MAX_RETRIES):
        try:
            return func(*args)
        except pynvml.NVMLError as err:
            code = err.value
            if code == pynvml.NVML_ERROR_GPU_IS_LOST:
                raise _translate_nvml_error(err) from err
            if code == pynvml.NVML_ERROR_UNKNOWN:
                last_err = err
                delay = _RETRY_BASE_DELAY * (2**attempt)
                func_name = getattr(func, "__name__", repr(func))
                logger.debug("NVML_ERROR_UNKNOWN on %s, retry %d/%d in %.1fs",
                             func_name, attempt + 1, _MAX_RETRIES, delay)
                time.sleep(delay)
            else:
                raise _translate_nvml_error(err) from err
    if last_err is None:
        msg = "Unreachable: _call_with_retry loop completed without error"
        raise DriverError(msg)
    func_name = getattr(func, "__name__", repr(func))
    raise DriverError(f"NVML call {func_name} failed after {_MAX_RETRIES} retries: {last_err}")


class NVMLClient(GPUBackend):
    """NVIDIA GPU backend using NVML via nvidia-ml-py."""

    def __init__(self) -> None:
        self._initialized = False

    def initialize(self) -> None:
        """Initialize NVML library."""
        if self._initialized:
            return
        try:
            pynvml.nvmlInit()
        except pynvml.NVMLError as err:
            raise BackendError(
                f"Failed to initialize NVML: {err}. "
                "Ensure NVIDIA drivers are installed."
            ) from err
        self._initialized = True

    def shutdown(self) -> None:
        """Shut down NVML library."""
        if not self._initialized:
            return
        with contextlib.suppress(pynvml.NVMLError):
            pynvml.nvmlShutdown()
        self._initialized = False

    def device_count(self) -> int:
        """Return the number of NVIDIA GPUs."""
        return _call_with_retry(pynvml.nvmlDeviceGetCount)  # type: ignore[no-any-return]

    def snapshot(self) -> MemorySnapshot:
        """Take a snapshot of all GPU state."""
        ts_mono = time.monotonic()
        ts_wall = time.time()

        try:
            driver_version = pynvml.nvmlSystemGetDriverVersion()
            nvml_version = pynvml.nvmlSystemGetNVMLVersion()
        except pynvml.NVMLError as err:
            raise _translate_nvml_error(err) from err

        count = self.device_count()
        devices: list[GPUDevice] = []

        for idx in range(count):
            try:
                dev = self._snapshot_device(idx)
                devices.append(dev)
            except GPULostError:
                logger.warning("GPU %d lost, skipping", idx)
                raise
            except DriverError:
                logger.warning("Driver error on GPU %d, skipping", idx)
                raise

        return MemorySnapshot(
            timestamp=ts_mono,
            wall_time=ts_wall,
            devices=tuple(devices),
            driver_version=driver_version,
            nvml_version=nvml_version,
        )

    def _snapshot_device(self, index: int) -> GPUDevice:
        """Snapshot a single GPU device."""
        handle = _call_with_retry(pynvml.nvmlDeviceGetHandleByIndex, index)

        name = _call_with_retry(pynvml.nvmlDeviceGetName, handle)
        uuid = _call_with_retry(pynvml.nvmlDeviceGetUUID, handle)

        # Use v2 memory info if available, fall back to v1
        mem_info = self._get_memory_info(handle)

        util = _call_with_retry(pynvml.nvmlDeviceGetUtilizationRates, handle)

        temp = _call_with_retry(
            pynvml.nvmlDeviceGetTemperature, handle, pynvml.NVML_TEMPERATURE_GPU
        )

        # Power in milliwatts → watts
        power_mw = _call_with_retry(pynvml.nvmlDeviceGetPowerUsage, handle)
        power_w = power_mw / 1000.0

        processes = self._get_merged_processes(handle)

        # Compute application-level used memory.
        # v1 nvmlDeviceGetMemoryInfo lumps driver-reserved memory into "used",
        # inflating it (e.g. 305 MiB on idle GPU).  v2 has a "reserved" field
        # but isn't on all drivers.  When v2 is missing, derive application
        # usage from process sum.  In both cases, mem_info.free is correct
        # (truly allocatable memory).
        reserved: int = getattr(mem_info, "reserved", 0)
        if reserved > 0:
            # v2 API: used already excludes reserved
            app_used: int = mem_info.used
        else:
            # v1 API: use process sum as application-level figure
            app_used = sum(p.used_memory_bytes for p in processes)

        return GPUDevice(
            index=index,
            uuid=uuid,
            name=name,
            total_memory_bytes=mem_info.total,
            used_memory_bytes=app_used,
            free_memory_bytes=mem_info.free,
            gpu_util_percent=util.gpu,
            mem_util_percent=util.memory,
            temperature_celsius=temp,
            power_watts=power_w,
            processes=tuple(processes),
        )

    @staticmethod
    def _get_memory_info(handle: Any) -> Any:
        """Get memory info, preferring v2 API with v1 fallback."""
        try:
            get_v2 = getattr(pynvml, "nvmlDeviceGetMemoryInfo_v2", None)
            if get_v2 is not None:
                return get_v2(handle)
        except pynvml.NVMLError:
            pass
        try:
            return pynvml.nvmlDeviceGetMemoryInfo(handle)
        except pynvml.NVMLError as err:
            raise _translate_nvml_error(err) from err

    @staticmethod
    def _get_merged_processes(handle: Any) -> list[GPUProcess]:
        """Get compute + graphics processes, merged/deduped by PID.

        A process appearing in both lists gets process_type="compute+graphics"
        and its memory is summed from both entries.
        """
        compute_procs: list[Any] = []
        graphics_procs: list[Any] = []

        try:
            compute_procs = _call_with_retry(
                pynvml.nvmlDeviceGetComputeRunningProcesses, handle
            )
        except (BackendError, pynvml.NVMLError):
            logger.debug("Failed to get compute processes")

        try:
            graphics_procs = _call_with_retry(
                pynvml.nvmlDeviceGetGraphicsRunningProcesses, handle
            )
        except (BackendError, pynvml.NVMLError):
            logger.debug("Failed to get graphics processes")

        # Merge by PID: track memory and type per PID.
        # IMPORTANT: A process appearing in BOTH lists reports the SAME
        # usedGpuMemory in each — it is NOT additive. Use max(), not sum().
        pid_memory: dict[int, int] = {}
        pid_types: dict[int, set[str]] = {}

        for proc in compute_procs:
            pid = proc.pid
            mem = proc.usedGpuMemory or 0
            pid_memory[pid] = max(pid_memory.get(pid, 0), mem)
            pid_types.setdefault(pid, set()).add("compute")

        for proc in graphics_procs:
            pid = proc.pid
            mem = proc.usedGpuMemory or 0
            pid_memory[pid] = max(pid_memory.get(pid, 0), mem)
            pid_types.setdefault(pid, set()).add("graphics")

        result: list[GPUProcess] = []
        for pid in pid_memory:
            types = pid_types[pid]
            if "compute" in types and "graphics" in types:
                ptype = "compute+graphics"
            elif "compute" in types:
                ptype = "compute"
            else:
                ptype = "graphics"

            identity = _get_identity(pid)
            name = _resolve_process_name(pid)

            result.append(
                GPUProcess(
                    identity=identity,
                    name=name,
                    used_memory_bytes=pid_memory[pid],
                    process_type=ptype,
                )
            )

        return result
