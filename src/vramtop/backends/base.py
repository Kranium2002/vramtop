"""Core data types, backend ABC, and exception hierarchy."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

# --- Exceptions ---


class VramtopError(Exception):
    """Base exception for all vramtop errors."""


class BackendError(VramtopError):
    """GPU backend initialization or communication failure."""


class GPULostError(BackendError):
    """GPU fell off the bus (NVML_ERROR_GPU_IS_LOST)."""


class DriverError(BackendError):
    """Driver crash or unknown NVML error after retries."""


class ConfigError(VramtopError):
    """Configuration loading or validation failure."""


# --- Data Types (frozen, slotted) ---


@dataclass(frozen=True, slots=True)
class ProcessIdentity:
    """Unique process key resistant to PID recycling."""

    pid: int
    starttime: int


@dataclass(frozen=True, slots=True)
class GPUProcess:
    """A single process using GPU memory."""

    identity: ProcessIdentity
    name: str
    used_memory_bytes: int
    process_type: str  # "compute", "graphics", or "compute+graphics"


@dataclass(frozen=True, slots=True)
class GPUDevice:
    """Snapshot of a single GPU's state."""

    index: int
    uuid: str
    name: str
    total_memory_bytes: int
    used_memory_bytes: int
    free_memory_bytes: int
    gpu_util_percent: int
    mem_util_percent: int
    temperature_celsius: int
    power_watts: float
    processes: Sequence[GPUProcess] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class MemorySnapshot:
    """Complete snapshot of all GPUs at a point in time."""

    timestamp: float  # time.monotonic()
    wall_time: float  # time.time()
    devices: Sequence[GPUDevice]
    driver_version: str
    nvml_version: str


# --- Backend ABC ---


class GPUBackend(ABC):
    """Abstract base class for GPU monitoring backends."""

    @abstractmethod
    def initialize(self) -> None:
        """Initialize the backend (e.g., nvmlInit)."""

    @abstractmethod
    def shutdown(self) -> None:
        """Clean up backend resources (e.g., nvmlShutdown)."""

    @abstractmethod
    def snapshot(self) -> MemorySnapshot:
        """Take a snapshot of all GPU state."""

    @abstractmethod
    def device_count(self) -> int:
        """Return the number of GPUs."""

    def __enter__(self) -> GPUBackend:
        self.initialize()
        return self

    def __exit__(self, *args: object) -> None:
        self.shutdown()
