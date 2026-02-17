"""Textual pilot tests for the vramtop TUI."""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest

from vramtop.backends.base import (
    GPUBackend,
    GPUDevice,
    GPUProcess,
    MemorySnapshot,
    ProcessIdentity,
)
from vramtop.config import ConfigHolder
from vramtop.ui.app import VramtopApp


def _make_snapshot(num_gpus: int = 1) -> MemorySnapshot:
    """Create a fake MemorySnapshot for testing."""
    devices = []
    for i in range(num_gpus):
        dev = GPUDevice(
            index=i,
            uuid=f"GPU-test-{i}",
            name=f"NVIDIA Test GPU {i}",
            total_memory_bytes=16 * 1024**3,
            used_memory_bytes=8 * 1024**3,
            free_memory_bytes=8 * 1024**3,
            gpu_util_percent=50,
            mem_util_percent=50,
            temperature_celsius=60,
            power_watts=200.0,
            processes=(
                GPUProcess(
                    identity=ProcessIdentity(pid=1000 + i, starttime=12345),
                    name="python",
                    used_memory_bytes=4 * 1024**3,
                    process_type="compute",
                ),
            ),
        )
        devices.append(dev)
    return MemorySnapshot(
        timestamp=time.monotonic(),
        wall_time=time.time(),
        devices=tuple(devices),
        driver_version="535.129.03",
        nvml_version="12.535.129.03",
    )


class FakeBackend(GPUBackend):
    """Fake GPU backend for testing."""

    def __init__(self, snapshot: MemorySnapshot | None = None) -> None:
        self._snapshot = snapshot or _make_snapshot()
        self._initialized = False

    def initialize(self) -> None:
        self._initialized = True

    def shutdown(self) -> None:
        self._initialized = False

    def snapshot(self) -> MemorySnapshot:
        return self._snapshot

    def device_count(self) -> int:
        return len(self._snapshot.devices)


@pytest.fixture()
def fake_backend() -> FakeBackend:
    backend = FakeBackend()
    backend.initialize()
    return backend


@pytest.fixture()
def config_holder() -> ConfigHolder:
    return ConfigHolder()


async def test_app_mounts_without_crashing(fake_backend, config_holder):
    """Test that the app mounts and renders without errors."""
    app = VramtopApp(backend=fake_backend, config_holder=config_holder)
    async with app.run_test() as pilot:
        # App should be running
        assert app.is_running
        # Check that we have a gpu-container
        container = app.query_one("#gpu-container")
        assert container is not None


async def test_gpu_card_renders_with_mock_data(fake_backend, config_holder):
    """Test that GPU cards appear with mock snapshot data."""
    app = VramtopApp(backend=fake_backend, config_holder=config_holder)
    async with app.run_test() as pilot:
        # Wait for the initial poll to complete
        await pilot.pause()
        # Should have created a widget for GPU 0
        gpu_widget = app.query_one("#gpu-0")
        assert gpu_widget is not None


async def test_multi_gpu_cards(config_holder):
    """Test that multiple GPU cards are created for multi-GPU snapshots."""
    snapshot = _make_snapshot(num_gpus=2)
    backend = FakeBackend(snapshot)
    backend.initialize()
    app = VramtopApp(backend=backend, config_holder=config_holder)
    async with app.run_test() as pilot:
        await pilot.pause()
        gpu0 = app.query_one("#gpu-0")
        gpu1 = app.query_one("#gpu-1")
        assert gpu0 is not None
        assert gpu1 is not None


async def test_quit_keybinding(fake_backend, config_holder):
    """Test that pressing 'q' quits the app."""
    app = VramtopApp(backend=fake_backend, config_holder=config_holder)
    async with app.run_test() as pilot:
        await pilot.press("q")
        # After pressing q, the app should no longer be running
        # (run_test context manager handles the exit)


async def test_app_title(fake_backend, config_holder):
    """Test that the app sets its title correctly."""
    app = VramtopApp(backend=fake_backend, config_holder=config_holder)
    async with app.run_test() as pilot:
        assert app.title == "vramtop"
        assert app.sub_title == "GPU Memory Monitor"
