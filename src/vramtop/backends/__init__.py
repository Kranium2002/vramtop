"""GPU backend abstraction and factory."""

from __future__ import annotations

from vramtop.backends.base import (
    BackendError,
    ConfigError,
    DriverError,
    GPUBackend,
    GPUDevice,
    GPULostError,
    GPUProcess,
    MemorySnapshot,
    ProcessIdentity,
    VramtopError,
)

__all__ = [
    "BackendError",
    "ConfigError",
    "DriverError",
    "GPUBackend",
    "GPUDevice",
    "GPULostError",
    "GPUProcess",
    "MemorySnapshot",
    "ProcessIdentity",
    "VramtopError",
    "get_backend",
]


def get_backend() -> GPUBackend:
    """Create and return the appropriate GPU backend.

    Currently only supports NVIDIA via NVML.
    """
    from vramtop.backends.nvidia import NVMLClient

    return NVMLClient()
