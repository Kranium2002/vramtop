"""Fake NVML response objects for testing the NVIDIA backend."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FakeNVMLProcess:
    """Mimics pynvml process info struct."""

    pid: int
    usedGpuMemory: int  # noqa: N815


@dataclass
class FakeMemoryInfo:
    """Mimics pynvml memory info struct."""

    total: int
    used: int
    free: int


@dataclass
class FakeUtilization:
    """Mimics pynvml utilization rates struct."""

    gpu: int
    memory: int


@dataclass
class FakeDeviceHandle:
    """Opaque handle representing a GPU device in tests."""

    index: int


@dataclass
class FakeGPUScenario:
    """A complete fake GPU configuration for testing."""

    handle: FakeDeviceHandle
    name: str
    uuid: str
    memory: FakeMemoryInfo
    utilization: FakeUtilization
    temperature: int
    power_mw: int  # milliwatts
    compute_processes: list[FakeNVMLProcess] = field(default_factory=list)
    graphics_processes: list[FakeNVMLProcess] = field(default_factory=list)


# --- Pre-built scenarios ---

SINGLE_GPU_COMPUTE = FakeGPUScenario(
    handle=FakeDeviceHandle(index=0),
    name="NVIDIA A100-SXM4-80GB",
    uuid="GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
    memory=FakeMemoryInfo(total=85899345920, used=42949672960, free=42949672960),
    utilization=FakeUtilization(gpu=75, memory=60),
    temperature=65,
    power_mw=250000,
    compute_processes=[
        FakeNVMLProcess(pid=1001, usedGpuMemory=20_000_000_000),
        FakeNVMLProcess(pid=1002, usedGpuMemory=10_000_000_000),
    ],
    graphics_processes=[],
)

SINGLE_GPU_MIXED = FakeGPUScenario(
    handle=FakeDeviceHandle(index=0),
    name="NVIDIA RTX 4090",
    uuid="GPU-11111111-2222-3333-4444-555555555555",
    memory=FakeMemoryInfo(total=25769803776, used=12884901888, free=12884901888),
    utilization=FakeUtilization(gpu=50, memory=40),
    temperature=70,
    power_mw=350000,
    compute_processes=[
        FakeNVMLProcess(pid=2001, usedGpuMemory=4_000_000_000),
    ],
    graphics_processes=[
        FakeNVMLProcess(pid=2001, usedGpuMemory=1_000_000_000),  # same PID as compute
        FakeNVMLProcess(pid=2002, usedGpuMemory=2_000_000_000),
    ],
)

MULTI_GPU = [
    FakeGPUScenario(
        handle=FakeDeviceHandle(index=0),
        name="NVIDIA A100-SXM4-80GB",
        uuid="GPU-aaaaaaaa-0000-0000-0000-000000000000",
        memory=FakeMemoryInfo(total=85899345920, used=42949672960, free=42949672960),
        utilization=FakeUtilization(gpu=90, memory=85),
        temperature=72,
        power_mw=300000,
        compute_processes=[
            FakeNVMLProcess(pid=3001, usedGpuMemory=40_000_000_000),
        ],
        graphics_processes=[],
    ),
    FakeGPUScenario(
        handle=FakeDeviceHandle(index=1),
        name="NVIDIA A100-SXM4-80GB",
        uuid="GPU-bbbbbbbb-0000-0000-0000-000000000000",
        memory=FakeMemoryInfo(total=85899345920, used=10737418240, free=75161927680),
        utilization=FakeUtilization(gpu=20, memory=15),
        temperature=55,
        power_mw=150000,
        compute_processes=[
            FakeNVMLProcess(pid=3002, usedGpuMemory=8_000_000_000),
        ],
        graphics_processes=[],
    ),
]

GPU_LOST_SCENARIO = FakeGPUScenario(
    handle=FakeDeviceHandle(index=0),
    name="NVIDIA A100-SXM4-80GB",
    uuid="GPU-dead0000-0000-0000-0000-000000000000",
    memory=FakeMemoryInfo(total=0, used=0, free=0),
    utilization=FakeUtilization(gpu=0, memory=0),
    temperature=0,
    power_mw=0,
)

EMPTY_GPU = FakeGPUScenario(
    handle=FakeDeviceHandle(index=0),
    name="NVIDIA T4",
    uuid="GPU-cccccccc-0000-0000-0000-000000000000",
    memory=FakeMemoryInfo(total=16106127360, used=0, free=16106127360),
    utilization=FakeUtilization(gpu=0, memory=0),
    temperature=35,
    power_mw=20000,
    compute_processes=[],
    graphics_processes=[],
)
