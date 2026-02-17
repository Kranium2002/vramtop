"""Real-world scenario tests for segment labeling.

These tests simulate realistic GPU workload patterns (training, inference,
multi-tenant, etc.) and verify the correct sequence of labels is assigned.
Each test represents a complete lifecycle that a real user would see.
"""

from __future__ import annotations

import math

from vramtop.analysis.phase_detector import Phase
from vramtop.analysis.segment_labels import (
    SegmentLabel,
    compute_segment_stats,
)

# ---------------------------------------------------------------------------
# Timeseries builders for realistic patterns
# ---------------------------------------------------------------------------


def _flat(value: float, n: int) -> list[float]:
    return [value] * n


def _ramp(start: float, end: float, n: int) -> list[float]:
    step = (end - start) / max(n - 1, 1)
    return [start + i * step for i in range(n)]


def _sine(center: float, amp: float, n: int, freq: float = 1.5) -> list[float]:
    return [center + amp * math.sin(i * freq) for i in range(n)]


def _staircase(start: float, step_size: float, steps: int, flat_per: int) -> list[float]:
    ts: list[float] = []
    for s in range(steps):
        ts.extend([start + s * step_size] * flat_per)
    return ts


def _micro_jitter(value: float, jitter: float, n: int) -> list[float]:
    """Stable with tiny sine jitter (for active inference)."""
    return [value + jitter * math.sin(i * 0.3) for i in range(n)]


# ---------------------------------------------------------------------------
# Scenario 1: PyTorch Training (full lifecycle)
# ---------------------------------------------------------------------------


class TestPyTorchTraining:
    """Simulates: model load → optimizer init → batch training → checkpoint →
    more training → cleanup."""

    def test_full_training_lifecycle(self) -> None:
        # Phase 1: Model loading (0→4000 MB, rapid)
        init = _ramp(0.0, 4000.0, 15)
        # Phase 2: Optimizer states (growth after init -> Optimizer Init)
        opt_init = _ramp(4000.0, 5200.0, 12)
        # Phase 3: Steady state before training starts
        steady = _flat(5200.0, 20)
        # Phase 4: Batch processing epoch 1
        batch1 = _sine(5200.0, 300.0, 40)
        # Phase 5: Epoch boundary (short volatile between batch segments)
        epoch_gap = _sine(5100.0, 200.0, 8)
        # Phase 6: Batch processing epoch 2
        batch2 = _sine(5200.0, 300.0, 40)
        # Phase 7: Cleanup
        cleanup = _ramp(5200.0, 200.0, 10)
        # Phase 8: Idle
        idle = _flat(5.0, 20)

        ts = (
            init + opt_init + steady + batch1
            + epoch_gap + batch2 + cleanup + idle
        )
        phases = [
            Phase.GROWING,   # init
            Phase.GROWING,   # opt_init
            Phase.STABLE,    # steady
            Phase.VOLATILE,  # batch1
            Phase.VOLATILE,  # epoch_gap
            Phase.VOLATILE,  # batch2
            Phase.SHRINKING, # cleanup
            Phase.STABLE,    # idle
        ]
        changepoints = [15, 27, 47, 87, 95, 135, 145]

        segments = compute_segment_stats(ts, changepoints, phases)
        assert len(segments) == 8

        assert segments[0].label == SegmentLabel.INITIALIZATION
        assert segments[1].label == SegmentLabel.OPTIMIZER_INIT
        assert segments[2].label == SegmentLabel.STEADY_STATE
        assert segments[3].label == SegmentLabel.BATCH_PROCESSING
        assert segments[4].label == SegmentLabel.EPOCH_BOUNDARY
        assert segments[5].label == SegmentLabel.BATCH_PROCESSING
        assert segments[6].label == SegmentLabel.CLEANUP
        assert segments[7].label == SegmentLabel.IDLE

    def test_training_with_gradient_accumulation(self) -> None:
        """Training with visible gradient accumulation steps.

        Gradient Steps only fires for non-early segments (position > 25%)
        to avoid false positives from pre-allocation / model loading.
        A stable warmup phase pushes the staircase past the 25% mark.
        """
        init = _ramp(0.0, 2000.0, 10)
        warmup_steady = _flat(2000.0, 20)
        grad_steps = _staircase(2000.0, 100.0, 8, 5)
        steady = _flat(2800.0, 30)
        cleanup = _ramp(2800.0, 100.0, 10)

        ts = init + warmup_steady + grad_steps + steady + cleanup
        phases = [
            Phase.GROWING,   # init
            Phase.STABLE,    # warmup_steady
            Phase.GROWING,   # grad_steps
            Phase.STABLE,    # steady
            Phase.SHRINKING, # cleanup
        ]
        changepoints = [10, 30, 70, 100]

        segments = compute_segment_stats(ts, changepoints, phases)
        assert len(segments) == 5
        assert segments[0].label == SegmentLabel.INITIALIZATION
        assert segments[2].label == SegmentLabel.GRADIENT_STEPS
        assert segments[3].label == SegmentLabel.STEADY_STATE
        assert segments[4].label == SegmentLabel.CLEANUP

    def test_training_with_memory_leak(self) -> None:
        """Training that develops a memory leak mid-run."""
        init = _ramp(0.0, 3000.0, 15)
        steady = _flat(3000.0, 40)
        batch = _sine(3000.0, 200.0, 60)
        # Late sustained growth = memory leak
        leak = _ramp(3200.0, 4500.0, 50)

        ts = init + steady + batch + leak
        phases = [
            Phase.GROWING,   # init
            Phase.STABLE,    # steady
            Phase.VOLATILE,  # batch
            Phase.GROWING,   # leak
        ]
        changepoints = [15, 55, 115]

        segments = compute_segment_stats(ts, changepoints, phases)
        assert len(segments) == 4
        assert segments[0].label == SegmentLabel.INITIALIZATION
        assert segments[1].label == SegmentLabel.STEADY_STATE
        assert segments[2].label == SegmentLabel.BATCH_PROCESSING
        assert segments[3].label == SegmentLabel.MEMORY_LEAK


# ---------------------------------------------------------------------------
# Scenario 2: Inference Server (vLLM / TGI style)
# ---------------------------------------------------------------------------


class TestInferenceServer:
    """Simulates: model load → cache fill → active inference → request bursts
    → cache eviction → steady."""

    def test_inference_with_kv_cache(self) -> None:
        # Load model
        init = _ramp(0.0, 8000.0, 10)
        # Steady state after model load (long enough to push cache past 0.25)
        steady = _flat(8000.0, 50)
        # KV cache fills gradually (position 60/220 = 0.27 → not early)
        cache_fill = _ramp(8000.0, 8100.0, 50)
        # Active inference: stable with micro-jitter
        active = _micro_jitter(8100.0, 5.0, 50)
        # Request burst
        burst = _sine(8200.0, 150.0, 10)
        # Back to active inference
        active2 = _micro_jitter(8100.0, 5.0, 50)

        ts = init + steady + cache_fill + active + burst + active2
        phases = [
            Phase.GROWING,   # init
            Phase.STABLE,    # steady
            Phase.GROWING,   # cache_fill
            Phase.STABLE,    # active
            Phase.VOLATILE,  # burst
            Phase.STABLE,    # active2
        ]
        changepoints = [10, 60, 110, 160, 170]

        segments = compute_segment_stats(ts, changepoints, phases)
        assert len(segments) == 6
        assert segments[0].label == SegmentLabel.INITIALIZATION
        assert segments[1].label == SegmentLabel.STEADY_STATE
        assert segments[2].label == SegmentLabel.CACHE_FILLING
        assert segments[3].label == SegmentLabel.ACTIVE_INFERENCE
        assert segments[4].label == SegmentLabel.REQUEST_BURST
        assert segments[5].label == SegmentLabel.ACTIVE_INFERENCE

    def test_inference_near_saturation(self) -> None:
        """Inference server running near GPU memory limit."""
        init = _ramp(0.0, 14000.0, 10)
        # Running at 90% GPU memory
        saturated = _flat(14400.0, 40)

        ts = init + saturated
        phases = [Phase.GROWING, Phase.STABLE]
        changepoints = [10]

        segments = compute_segment_stats(
            ts, changepoints, phases, gpu_total_mb=16000.0,
        )
        assert len(segments) == 2
        assert segments[0].label == SegmentLabel.INITIALIZATION
        assert segments[1].label == SegmentLabel.SATURATION

    def test_inference_with_cache_eviction(self) -> None:
        """Inference server evicting KV cache entries."""
        active = _micro_jitter(10000.0, 5.0, 50)
        # Moderate cache eviction (not too large, otherwise it's CLEANUP)
        evict = _ramp(10000.0, 9850.0, 15)
        # Stabilize at lower level
        active2 = _flat(9850.0, 30)

        ts = active + evict + active2
        phases = [Phase.STABLE, Phase.SHRINKING, Phase.STABLE]
        changepoints = [50, 65]

        segments = compute_segment_stats(ts, changepoints, phases)
        assert len(segments) == 3
        assert segments[0].label == SegmentLabel.ACTIVE_INFERENCE
        assert segments[1].label == SegmentLabel.CACHE_EVICTION
        assert segments[2].label == SegmentLabel.STEADY_STATE


# ---------------------------------------------------------------------------
# Scenario 3: Multi-GPU / Memory Pressure
# ---------------------------------------------------------------------------


class TestMemoryPressure:
    """Simulates workloads that approach GPU memory limits."""

    def test_growth_under_pressure(self) -> None:
        """Growing while already near GPU limit -> Memory Pressure."""
        init = _ramp(0.0, 12000.0, 15)
        steady = _flat(12000.0, 30)
        # Growth at 75%+ of 16GB GPU
        pressure = _ramp(12000.0, 14000.0, 20)

        ts = init + steady + pressure
        phases = [Phase.GROWING, Phase.STABLE, Phase.GROWING]
        changepoints = [15, 45]

        segments = compute_segment_stats(
            ts, changepoints, phases, gpu_total_mb=16000.0,
        )
        assert len(segments) == 3
        assert segments[0].label == SegmentLabel.INITIALIZATION
        assert segments[1].label == SegmentLabel.STEADY_STATE
        assert segments[2].label == SegmentLabel.MEMORY_PRESSURE

    def test_saturation_then_cleanup(self) -> None:
        """Process that saturates GPU then cleans up."""
        init = _ramp(0.0, 14000.0, 10)
        saturated = _flat(14000.0, 30)
        cleanup = _ramp(14000.0, 2000.0, 10)
        idle = _flat(5.0, 20)

        ts = init + saturated + cleanup + idle
        phases = [Phase.GROWING, Phase.STABLE, Phase.SHRINKING, Phase.STABLE]
        changepoints = [10, 40, 50]

        segments = compute_segment_stats(
            ts, changepoints, phases, gpu_total_mb=16000.0,
        )
        assert len(segments) == 4
        assert segments[0].label == SegmentLabel.INITIALIZATION
        assert segments[1].label == SegmentLabel.SATURATION
        assert segments[2].label == SegmentLabel.CLEANUP
        assert segments[3].label == SegmentLabel.IDLE


# ---------------------------------------------------------------------------
# Scenario 4: GC / Fragmentation patterns
# ---------------------------------------------------------------------------


class TestGCAndFragmentation:
    """Simulates garbage collection and memory fragmentation."""

    def test_gc_during_training(self) -> None:
        """Brief GC collection during steady training."""
        steady = _flat(5000.0, 30)
        gc_drop = _ramp(5000.0, 4970.0, 5)  # small delta < 50
        recovery = _ramp(4970.0, 5010.0, 10)
        steady2 = _flat(5000.0, 30)

        ts = steady + gc_drop + recovery + steady2
        phases = [
            Phase.STABLE, Phase.SHRINKING, Phase.GROWING, Phase.STABLE,
        ]
        changepoints = [30, 35, 45]

        segments = compute_segment_stats(ts, changepoints, phases)
        assert len(segments) == 4
        assert segments[0].label == SegmentLabel.STEADY_STATE
        assert segments[1].label == SegmentLabel.GC_COLLECTION

    def test_fragmentation_pattern(self) -> None:
        """Small oscillations with no net change -> Fragmentation."""
        # Tiny oscillations: delta < 20 and low variance
        ts = [4000.0 + (i % 3) * 3.0 for i in range(40)]

        segments = compute_segment_stats(ts, [], [Phase.VOLATILE])
        assert len(segments) == 1
        assert segments[0].label == SegmentLabel.FRAGMENTATION

    def test_allocation_event_spike(self) -> None:
        """Short burst of allocation after stable period."""
        stable = _flat(2000.0, 40)
        burst = _ramp(2000.0, 2200.0, 5)

        ts = stable + burst
        phases = [Phase.STABLE, Phase.GROWING]
        changepoints = [40]

        segments = compute_segment_stats(ts, changepoints, phases)
        assert len(segments) == 2
        assert segments[0].label == SegmentLabel.STEADY_STATE
        assert segments[1].label == SegmentLabel.ALLOCATION_EVENT


# ---------------------------------------------------------------------------
# Scenario 5: Multi-epoch training (complex lifecycle)
# ---------------------------------------------------------------------------


class TestMultiEpochTraining:
    """Simulates a multi-epoch training run with all the transitions."""

    def test_three_epoch_training(self) -> None:
        """Init -> Batch -> Epoch -> Batch -> Epoch -> Batch -> Cooldown."""
        init = _ramp(0.0, 3000.0, 15)
        batch1 = _sine(3000.0, 200.0, 40)
        epoch1 = _sine(2900.0, 100.0, 8)
        batch2 = _sine(3000.0, 200.0, 40)
        epoch2 = _sine(2900.0, 100.0, 8)
        batch3 = _sine(3000.0, 200.0, 40)
        release = _ramp(3000.0, 2800.0, 20)

        ts = init + batch1 + epoch1 + batch2 + epoch2 + batch3 + release
        phases = [
            Phase.GROWING,   # init
            Phase.VOLATILE,  # batch1
            Phase.VOLATILE,  # epoch1
            Phase.VOLATILE,  # batch2
            Phase.VOLATILE,  # epoch2
            Phase.VOLATILE,  # batch3
            Phase.SHRINKING, # release
        ]
        changepoints = [15, 55, 63, 103, 111, 151]

        segments = compute_segment_stats(ts, changepoints, phases)
        assert len(segments) == 7
        assert segments[0].label == SegmentLabel.INITIALIZATION
        assert segments[1].label == SegmentLabel.BATCH_PROCESSING
        assert segments[2].label == SegmentLabel.EPOCH_BOUNDARY
        assert segments[3].label == SegmentLabel.BATCH_PROCESSING
        assert segments[4].label == SegmentLabel.EPOCH_BOUNDARY
        assert segments[5].label == SegmentLabel.BATCH_PROCESSING
        assert segments[6].label == SegmentLabel.COOLDOWN


# ---------------------------------------------------------------------------
# Scenario 6: Short-lived process
# ---------------------------------------------------------------------------


class TestShortProcess:
    """Quick process that doesn't run long enough for complex patterns."""

    def test_quick_inference(self) -> None:
        """Load, run briefly, exit."""
        init = _ramp(0.0, 500.0, 5)
        flat = _flat(500.0, 3)

        ts = init + flat
        phases = [Phase.GROWING, Phase.STABLE]
        changepoints = [5]

        segments = compute_segment_stats(ts, changepoints, phases)
        assert len(segments) == 2
        assert segments[0].label == SegmentLabel.PRE_ALLOCATION
        assert segments[1].label == SegmentLabel.PLATEAU

    def test_burst_and_release(self) -> None:
        """Single allocation event then release."""
        stable = _flat(100.0, 40)
        burst = _ramp(100.0, 250.0, 5)
        release = _ramp(250.0, 100.0, 15)

        ts = stable + burst + release
        phases = [Phase.STABLE, Phase.GROWING, Phase.SHRINKING]
        changepoints = [40, 45]

        segments = compute_segment_stats(ts, changepoints, phases)
        assert len(segments) == 3
        assert segments[0].label == SegmentLabel.STEADY_STATE
        assert segments[1].label == SegmentLabel.ALLOCATION_EVENT
        assert segments[2].label == SegmentLabel.RELEASING


# ---------------------------------------------------------------------------
# Coverage: verify all 25 labels are testable
# ---------------------------------------------------------------------------


class TestLabelCoverage:
    """Ensure every label in the enum is tested at least once across
    all test classes in this file."""

    def test_all_25_labels_are_covered(self) -> None:
        """Construct scenarios that produce each of the 25 labels."""
        seen: set[SegmentLabel] = set()

        # Growth patterns
        # Initialization: early + very large
        seen.add(_get_label(_ramp(0.0, 500.0, 15), Phase.GROWING))
        # Pre-allocation: very early + very large + short
        seen.add(_get_label(
            _ramp(0.0, 2000.0, 5), Phase.GROWING,
        ))
        # Warmup: early + large (but not very large)
        seen.add(_get_label(_ramp(100.0, 200.0, 15), Phase.GROWING))
        # Allocation Event: short + large (not early)
        stable_prefix = _flat(100.0, 50)
        burst_ts = _ramp(100.0, 300.0, 5)
        segs = compute_segment_stats(
            stable_prefix + burst_ts, [50],
            [Phase.STABLE, Phase.GROWING],
        )
        seen.add(segs[1].label)  # ALLOCATION_EVENT
        # Memory Growth: late, not long
        stable2 = _flat(200.0, 60)
        growth = _ramp(200.0, 300.0, 20)
        segs = compute_segment_stats(
            stable2 + growth, [60],
            [Phase.STABLE, Phase.GROWING],
        )
        seen.add(segs[1].label)  # MEMORY_GROWTH
        # Memory Leak: late + long
        stable3 = _flat(200.0, 100)
        leak = _ramp(200.0, 350.0, 50)
        segs = compute_segment_stats(
            stable3 + leak, [100],
            [Phase.STABLE, Phase.GROWING],
        )
        seen.add(segs[1].label)  # MEMORY_LEAK
        # Cache Filling: mid-timeline + long
        early_s = _flat(100.0, 25)
        cache_g = _ramp(100.0, 200.0, 50)
        segs = compute_segment_stats(
            early_s + cache_g, [25],
            [Phase.STABLE, Phase.GROWING],
        )
        seen.add(segs[1].label)  # CACHE_FILLING
        # Gradient Steps: staircase (must be non-early, > 25% position)
        stable_pre = _flat(100.0, 20)
        staircase_ts = _staircase(100.0, 50.0, 8, 5)
        segs = compute_segment_stats(
            stable_pre + staircase_ts, [20],
            [Phase.STABLE, Phase.GROWING],
        )
        seen.add(segs[1].label)  # GRADIENT_STEPS
        # Memory Pressure: growing at >70% GPU
        seen.add(_get_label(
            _ramp(7500.0, 8500.0, 20), Phase.GROWING,
            gpu_total_mb=10000.0,
        ))
        # Optimizer Init: tested via multi-segment
        seen.add(SegmentLabel.OPTIMIZER_INIT)

        # Stable patterns
        seen.add(_get_label(_flat(100.0, 30), Phase.STABLE))
        seen.add(_get_label(
            _flat(9000.0, 30), Phase.STABLE, gpu_total_mb=10000.0,
        ))
        seen.add(_get_label(_flat(100.0, 3), Phase.STABLE))
        seen.add(_get_label(_flat(5.0, 30), Phase.STABLE))
        seen.add(_get_label(
            _micro_jitter(500.0, 3.0, 50), Phase.STABLE,
        ))

        # Volatile patterns
        seen.add(_get_label(_sine(200.0, 80.0, 30), Phase.VOLATILE))
        seen.add(_get_label(
            [200.0 + (i % 3) * 3.0 for i in range(30)], Phase.VOLATILE,
        ))
        seen.add(_get_label(_sine(200.0, 50.0, 5), Phase.VOLATILE))
        # Epoch Boundary and Request Burst: tested via multi-segment
        seen.add(SegmentLabel.EPOCH_BOUNDARY)
        seen.add(SegmentLabel.REQUEST_BURST)

        # Shrinking patterns
        seen.add(_get_label(_ramp(500.0, 0.0, 15), Phase.SHRINKING))
        seen.add(_get_label(_ramp(200.0, 150.0, 15), Phase.SHRINKING))
        seen.add(_get_label(
            _ramp(200.0, 180.0, 50), Phase.SHRINKING,
        ))
        seen.add(_get_label(_ramp(200.0, 170.0, 5), Phase.SHRINKING))
        # Cache Eviction: tested via multi-segment
        seen.add(SegmentLabel.CACHE_EVICTION)

        assert len(seen) == 25, (
            f"Only {len(seen)}/25 labels covered: "
            f"missing {set(SegmentLabel) - seen}"
        )


def _get_label(
    ts: list[float],
    phase: Phase,
    total_len: int = 0,
    start_idx: int = 0,
    gpu_total_mb: float = 0.0,
) -> SegmentLabel:
    """Get the label assigned to a single-segment timeseries."""
    segments = compute_segment_stats(
        ts, [], [phase], gpu_total_mb=gpu_total_mb,
    )
    assert len(segments) == 1
    return segments[0].label
