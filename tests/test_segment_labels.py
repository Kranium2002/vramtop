"""Tests for model-agnostic segment labeling heuristics."""

from __future__ import annotations

import pytest

from vramtop.analysis.phase_detector import Phase
from vramtop.analysis.segment_labels import (
    LABEL_ARROW,
    LABEL_COLORS,
    LABEL_DESCRIPTIONS,
    LABEL_SHORT,
    SegmentLabel,
    compute_segment_stats,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _stable_ts(value: float = 100.0, length: int = 30) -> list[float]:
    """Flat timeseries at a constant value."""
    return [value] * length


def _growing_ts(start: float = 0.0, end: float = 500.0, length: int = 30) -> list[float]:
    """Linearly increasing timeseries."""
    step = (end - start) / max(length - 1, 1)
    return [start + i * step for i in range(length)]


def _shrinking_ts(start: float = 500.0, end: float = 0.0, length: int = 30) -> list[float]:
    """Linearly decreasing timeseries."""
    return _growing_ts(start, end, length)


def _volatile_ts(center: float = 200.0, amplitude: float = 50.0, length: int = 30) -> list[float]:
    """Saw-tooth timeseries simulating batch processing."""
    import math
    return [center + amplitude * math.sin(i * 1.5) for i in range(length)]


# ---------------------------------------------------------------------------
# Basic label assignment
# ---------------------------------------------------------------------------


class TestSegmentLabels:
    def test_single_stable_segment(self) -> None:
        ts = _stable_ts(100.0, 30)
        segments = compute_segment_stats(ts, [], [Phase.STABLE])
        assert len(segments) == 1
        assert segments[0].label == SegmentLabel.STEADY_STATE
        assert segments[0].phase == Phase.STABLE

    def test_single_growing_segment_early(self) -> None:
        """Early growing segment with large change -> Initialization."""
        ts = _growing_ts(0.0, 500.0, 30)
        segments = compute_segment_stats(ts, [], [Phase.GROWING])
        assert len(segments) == 1
        assert segments[0].label == SegmentLabel.INITIALIZATION

    def test_single_shrinking_large(self) -> None:
        """Large shrinking segment -> Cleanup."""
        ts = _shrinking_ts(500.0, 0.0, 30)
        segments = compute_segment_stats(ts, [], [Phase.SHRINKING])
        assert len(segments) == 1
        assert segments[0].label == SegmentLabel.CLEANUP

    def test_single_shrinking_small(self) -> None:
        """Small shrinking segment -> Releasing."""
        ts = _shrinking_ts(100.0, 80.0, 30)
        segments = compute_segment_stats(ts, [], [Phase.SHRINKING])
        assert len(segments) == 1
        assert segments[0].label == SegmentLabel.RELEASING

    def test_volatile_segment(self) -> None:
        """Volatile segment -> Batch Processing."""
        ts = _volatile_ts(200.0, 50.0, 30)
        segments = compute_segment_stats(ts, [], [Phase.VOLATILE])
        assert len(segments) == 1
        assert segments[0].label == SegmentLabel.BATCH_PROCESSING

    def test_idle_segment(self) -> None:
        """Stable at near-zero memory -> Idle."""
        ts = _stable_ts(5.0, 30)
        segments = compute_segment_stats(ts, [], [Phase.STABLE])
        assert len(segments) == 1
        assert segments[0].label == SegmentLabel.IDLE

    def test_pre_allocation(self) -> None:
        """Massive jump at very start, very short -> Pre-allocation."""
        burst = _growing_ts(0.0, 2000.0, 5)
        stable = _stable_ts(2000.0, 95)
        ts = burst + stable
        phases = [Phase.GROWING, Phase.STABLE]
        changepoints = [5]
        segments = compute_segment_stats(ts, changepoints, phases)
        assert segments[0].label == SegmentLabel.PRE_ALLOCATION

    def test_memory_leak(self) -> None:
        """Late, long, sustained growth -> Memory Leak."""
        stable = _stable_ts(200.0, 100)
        leak = _growing_ts(200.0, 350.0, 50)
        ts = stable + leak
        phases = [Phase.STABLE, Phase.GROWING]
        changepoints = [100]
        segments = compute_segment_stats(ts, changepoints, phases)
        # Position 100/150 = 0.67 -> is_late, duration 50 -> is_long
        assert segments[1].label == SegmentLabel.MEMORY_LEAK

    def test_saturation(self) -> None:
        """Stable at >85% GPU memory -> Saturation."""
        ts = _stable_ts(9000.0, 30)
        segments = compute_segment_stats(ts, [], [Phase.STABLE], gpu_total_mb=10000.0)
        assert len(segments) == 1
        assert segments[0].label == SegmentLabel.SATURATION

    def test_saturation_not_without_gpu_total(self) -> None:
        """Without gpu_total_mb, high stable -> Steady State, not Saturation."""
        ts = _stable_ts(9000.0, 30)
        segments = compute_segment_stats(ts, [], [Phase.STABLE])
        assert segments[0].label == SegmentLabel.STEADY_STATE

    def test_fragmentation(self) -> None:
        """Volatile with low net change, low variance -> Fragmentation."""
        # Small oscillations around a mean with minimal net delta
        ts = [200.0 + (i % 3) * 5.0 for i in range(30)]
        segments = compute_segment_stats(ts, [], [Phase.VOLATILE])
        assert len(segments) == 1
        assert segments[0].label == SegmentLabel.FRAGMENTATION

    def test_volatile_short_spike_is_checkpoint(self) -> None:
        """Very short volatile spike -> Checkpoint Save."""
        ts = [200.0, 250.0, 300.0, 280.0, 210.0]
        segments = compute_segment_stats(ts, [], [Phase.VOLATILE])
        assert len(segments) == 1
        assert segments[0].label == SegmentLabel.CHECKPOINT_SAVE

    def test_gradient_steps(self) -> None:
        """Staircase growth pattern after startup -> Gradient Steps.

        Gradient Steps only fires for non-early segments (position > 25%)
        to avoid false positives from pre-allocation / model loading.
        """
        # Prefix with a stable segment so the growing staircase starts
        # after position_frac > 0.25 (avoids Initialization label).
        stable_prefix: list[float] = [500.0] * 20
        staircase: list[float] = []
        for step in range(8):
            staircase.extend([500.0 + float(step * 50)] * 5)
        ts = stable_prefix + staircase
        # Changepoint at the boundary, two phases
        segments = compute_segment_stats(
            ts, [20], [Phase.STABLE, Phase.GROWING],
        )
        assert len(segments) == 2
        assert segments[1].label == SegmentLabel.GRADIENT_STEPS

    def test_cooldown_long_slow_shrink(self) -> None:
        """Long, slow shrinking -> Cooldown."""
        ts = _shrinking_ts(200.0, 180.0, 50)
        segments = compute_segment_stats(ts, [], [Phase.SHRINKING])
        assert len(segments) == 1
        assert segments[0].label == SegmentLabel.COOLDOWN

    def test_cache_filling_mid_growth(self) -> None:
        """Non-early, non-late long growth -> Cache Filling."""
        early_stable = _stable_ts(100.0, 25)
        growth = _growing_ts(100.0, 200.0, 50)
        ts = early_stable + growth
        phases = [Phase.STABLE, Phase.GROWING]
        changepoints = [25]
        segments = compute_segment_stats(ts, changepoints, phases)
        # Second segment starts at position 25/75 = 0.33 (not early, not late), long
        assert segments[1].label == SegmentLabel.CACHE_FILLING


# ---------------------------------------------------------------------------
# Multi-segment timeseries
# ---------------------------------------------------------------------------


class TestMultiSegment:
    def test_init_then_steady(self) -> None:
        """Growing -> Stable = Initialization -> Steady State."""
        growing = _growing_ts(0.0, 500.0, 20)
        stable = _stable_ts(500.0, 30)
        ts = growing + stable
        phases = [Phase.GROWING, Phase.STABLE]
        changepoints = [20]

        segments = compute_segment_stats(ts, changepoints, phases)
        assert len(segments) == 2
        assert segments[0].label == SegmentLabel.INITIALIZATION
        assert segments[1].label == SegmentLabel.STEADY_STATE

    def test_init_optimizer_steady(self) -> None:
        """Three-phase lifecycle: Init -> Optimizer Init -> Steady."""
        init = _growing_ts(0.0, 400.0, 10)
        optimizer = _growing_ts(400.0, 500.0, 15)
        steady = _stable_ts(500.0, 25)
        ts = init + optimizer + steady
        phases = [Phase.GROWING, Phase.GROWING, Phase.STABLE]
        changepoints = [10, 25]

        segments = compute_segment_stats(ts, changepoints, phases)
        assert len(segments) == 3
        assert segments[0].label == SegmentLabel.INITIALIZATION
        # Growth after initialization -> Optimizer Init (refined)
        assert segments[1].label == SegmentLabel.OPTIMIZER_INIT
        assert segments[2].label == SegmentLabel.STEADY_STATE

    def test_training_lifecycle(self) -> None:
        """Full training lifecycle: Init -> Batch Processing -> Cleanup."""
        init = _growing_ts(0.0, 1000.0, 15)
        batch = _volatile_ts(1000.0, 100.0, 50)
        cleanup = _shrinking_ts(1000.0, 100.0, 15)
        ts = init + batch + cleanup
        phases = [Phase.GROWING, Phase.VOLATILE, Phase.SHRINKING]
        changepoints = [15, 65]

        segments = compute_segment_stats(ts, changepoints, phases)
        assert len(segments) == 3
        assert segments[0].label == SegmentLabel.INITIALIZATION
        assert segments[1].label == SegmentLabel.BATCH_PROCESSING
        assert segments[2].label == SegmentLabel.CLEANUP

    def test_late_growing_is_memory_growth(self) -> None:
        """Growing segment late in the timeline -> Memory Growth."""
        stable = _stable_ts(200.0, 60)
        growing_late = _growing_ts(200.0, 400.0, 20)
        ts = stable + growing_late
        phases = [Phase.STABLE, Phase.GROWING]
        changepoints = [60]

        segments = compute_segment_stats(ts, changepoints, phases)
        assert len(segments) == 2
        assert segments[1].label == SegmentLabel.MEMORY_GROWTH


# ---------------------------------------------------------------------------
# Multi-segment refinement (second pass)
# ---------------------------------------------------------------------------


class TestMultiSegmentRefinement:
    def test_checkpoint_save_between_stable(self) -> None:
        """Volatile between two stable at same level -> Checkpoint Save."""
        stable1 = _stable_ts(500.0, 30)
        spike = _volatile_ts(500.0, 100.0, 10)
        stable2 = _stable_ts(500.0, 30)
        ts = stable1 + spike + stable2
        phases = [Phase.STABLE, Phase.VOLATILE, Phase.STABLE]
        changepoints = [30, 40]

        segments = compute_segment_stats(ts, changepoints, phases)
        assert len(segments) == 3
        assert segments[1].label == SegmentLabel.CHECKPOINT_SAVE

    def test_cooldown_after_volatile(self) -> None:
        """Shrinking after volatile -> Cooldown (refined from Releasing)."""
        batch = _volatile_ts(500.0, 100.0, 30)
        release = _shrinking_ts(500.0, 450.0, 20)
        ts = batch + release
        phases = [Phase.VOLATILE, Phase.SHRINKING]
        changepoints = [30]

        segments = compute_segment_stats(ts, changepoints, phases)
        assert len(segments) == 2
        assert segments[1].label == SegmentLabel.COOLDOWN

    def test_cache_filling_before_stable(self) -> None:
        """Growing segment followed by stable -> Cache Filling (refined)."""
        early_stable = _stable_ts(100.0, 30)
        growth = _growing_ts(100.0, 200.0, 20)
        stable_after = _stable_ts(200.0, 30)
        ts = early_stable + growth + stable_after
        phases = [Phase.STABLE, Phase.GROWING, Phase.STABLE]
        changepoints = [30, 50]

        segments = compute_segment_stats(ts, changepoints, phases)
        assert len(segments) == 3
        # Growing before stable with rising level -> Cache Filling
        assert segments[1].label == SegmentLabel.CACHE_FILLING


# ---------------------------------------------------------------------------
# Segment statistics
# ---------------------------------------------------------------------------


class TestSegmentStats:
    def test_segment_info_fields(self) -> None:
        ts = _growing_ts(100.0, 200.0, 20)
        segments = compute_segment_stats(ts, [], [Phase.GROWING])
        seg = segments[0]
        assert seg.start_idx == 0
        assert seg.end_idx == 20
        assert seg.duration_samples == 20
        assert seg.delta_mb == pytest.approx(100.0, abs=1.0)
        assert seg.mean_mb == pytest.approx(150.0, abs=5.0)
        assert seg.rate_mb_per_sample == pytest.approx(5.0, abs=1.0)

    def test_variance_is_computed(self) -> None:
        ts = _volatile_ts(200.0, 50.0, 30)
        segments = compute_segment_stats(ts, [], [Phase.VOLATILE])
        assert segments[0].variance > 0.0

    def test_stable_has_low_variance(self) -> None:
        ts = _stable_ts(100.0, 30)
        segments = compute_segment_stats(ts, [], [Phase.STABLE])
        assert segments[0].variance == 0.0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_timeseries(self) -> None:
        assert compute_segment_stats([], [], []) == []

    def test_empty_phases(self) -> None:
        assert compute_segment_stats([1.0, 2.0], [], []) == []

    def test_single_point_segment(self) -> None:
        """Single-point segment should not crash."""
        ts = [100.0, 200.0]
        segments = compute_segment_stats(ts, [1], [Phase.STABLE, Phase.STABLE])
        assert len(segments) == 2

    def test_short_burst_growing(self) -> None:
        """Short growing burst -> Allocation Event."""
        stable = _stable_ts(100.0, 50)
        burst = _growing_ts(100.0, 300.0, 5)
        ts = stable + burst
        phases = [Phase.STABLE, Phase.GROWING]
        changepoints = [50]
        segments = compute_segment_stats(ts, changepoints, phases)
        assert segments[1].label == SegmentLabel.ALLOCATION_EVENT

    def test_short_stable_is_plateau(self) -> None:
        """Very short stable segment -> Plateau."""
        growing = _growing_ts(0.0, 200.0, 50)
        flat = _stable_ts(200.0, 3)
        ts = growing + flat
        phases = [Phase.GROWING, Phase.STABLE]
        changepoints = [50]
        segments = compute_segment_stats(ts, changepoints, phases)
        assert segments[1].label == SegmentLabel.PLATEAU


# ---------------------------------------------------------------------------
# Label metadata
# ---------------------------------------------------------------------------


class TestLabelMetadata:
    def test_all_labels_have_colors(self) -> None:
        for label in SegmentLabel:
            assert label in LABEL_COLORS

    def test_all_labels_have_short_names(self) -> None:
        for label in SegmentLabel:
            assert label in LABEL_SHORT

    def test_short_names_are_short(self) -> None:
        for label in SegmentLabel:
            assert len(LABEL_SHORT[label]) <= 7

    def test_all_labels_have_arrows(self) -> None:
        for label in SegmentLabel:
            assert label in LABEL_ARROW

    def test_all_labels_have_descriptions(self) -> None:
        for label in SegmentLabel:
            assert label in LABEL_DESCRIPTIONS
            assert len(LABEL_DESCRIPTIONS[label]) > 5


# ---------------------------------------------------------------------------
# New internal labels (Pass 1)
# ---------------------------------------------------------------------------


class TestInternalLabelsPass1:
    def test_memory_pressure_growing_near_limit(self) -> None:
        """Growing at >70% GPU capacity -> Memory Pressure."""
        stable = _stable_ts(7500.0, 30)
        growth = _growing_ts(7500.0, 8500.0, 20)
        ts = stable + growth
        phases = [Phase.STABLE, Phase.GROWING]
        changepoints = [30]
        segments = compute_segment_stats(
            ts, changepoints, phases, gpu_total_mb=10000.0,
        )
        assert segments[1].label == SegmentLabel.MEMORY_PRESSURE

    def test_memory_pressure_requires_gpu_total(self) -> None:
        """Without gpu_total_mb, high growth is not Memory Pressure."""
        ts = _growing_ts(7500.0, 8500.0, 20)
        segments = compute_segment_stats(ts, [], [Phase.GROWING])
        assert segments[0].label != SegmentLabel.MEMORY_PRESSURE

    def test_active_inference_stable_with_micro_fluctuations(self) -> None:
        """Stable with tiny variance over long duration -> Active Inference."""
        import math
        # Simulate inference: stable ~500MB with tiny wobble
        ts = [500.0 + 2.0 * math.sin(i * 0.5) for i in range(50)]
        segments = compute_segment_stats(ts, [], [Phase.STABLE])
        assert len(segments) == 1
        assert segments[0].label == SegmentLabel.ACTIVE_INFERENCE

    def test_active_inference_not_flat(self) -> None:
        """Pure flat timeseries (variance=0) -> Steady State, not Active Inference."""
        ts = _stable_ts(500.0, 50)
        segments = compute_segment_stats(ts, [], [Phase.STABLE])
        assert segments[0].label == SegmentLabel.STEADY_STATE

    def test_gc_collection_short_shrink(self) -> None:
        """Very short shrinking with small delta -> GC Collection."""
        ts = _shrinking_ts(200.0, 170.0, 5)
        segments = compute_segment_stats(ts, [], [Phase.SHRINKING])
        assert len(segments) == 1
        assert segments[0].label == SegmentLabel.GC_COLLECTION


# ---------------------------------------------------------------------------
# New internal labels (Pass 2 refinement)
# ---------------------------------------------------------------------------


class TestInternalLabelsPass2:
    def test_optimizer_init_after_initialization(self) -> None:
        """Growth right after Initialization -> Optimizer Init."""
        init = _growing_ts(0.0, 1000.0, 15)
        opt = _growing_ts(1000.0, 1100.0, 12)
        steady = _stable_ts(1100.0, 30)
        ts = init + opt + steady
        phases = [Phase.GROWING, Phase.GROWING, Phase.STABLE]
        changepoints = [15, 27]
        segments = compute_segment_stats(ts, changepoints, phases)
        assert segments[0].label == SegmentLabel.INITIALIZATION
        assert segments[1].label == SegmentLabel.OPTIMIZER_INIT

    def test_epoch_boundary_between_batches(self) -> None:
        """Short volatile between two batch processing -> Epoch Boundary."""
        batch1 = _volatile_ts(500.0, 80.0, 30)
        gap = _volatile_ts(480.0, 40.0, 10)
        batch2 = _volatile_ts(500.0, 80.0, 30)
        ts = batch1 + gap + batch2
        phases = [Phase.VOLATILE, Phase.VOLATILE, Phase.VOLATILE]
        changepoints = [30, 40]
        segments = compute_segment_stats(ts, changepoints, phases)
        assert len(segments) == 3
        assert segments[0].label == SegmentLabel.BATCH_PROCESSING
        assert segments[1].label == SegmentLabel.EPOCH_BOUNDARY
        assert segments[2].label == SegmentLabel.BATCH_PROCESSING

    def test_request_burst_after_steady(self) -> None:
        """Short volatile after steady state -> Request Burst."""
        steady = _stable_ts(300.0, 40)
        burst = _volatile_ts(350.0, 80.0, 10)
        ts = steady + burst
        phases = [Phase.STABLE, Phase.VOLATILE]
        changepoints = [40]
        segments = compute_segment_stats(ts, changepoints, phases)
        assert segments[0].label == SegmentLabel.STEADY_STATE
        assert segments[1].label == SegmentLabel.REQUEST_BURST

    def test_cache_eviction_after_steady(self) -> None:
        """Moderate shrink after steady state -> Cache Eviction."""
        steady = _stable_ts(500.0, 30)
        shrink = _shrinking_ts(500.0, 400.0, 15)
        ts = steady + shrink
        phases = [Phase.STABLE, Phase.SHRINKING]
        changepoints = [30]
        segments = compute_segment_stats(ts, changepoints, phases)
        assert segments[0].label == SegmentLabel.STEADY_STATE
        assert segments[1].label == SegmentLabel.CACHE_EVICTION

    def test_gc_collection_before_growth(self) -> None:
        """Short shrink followed by growth -> GC Collection (confirmed)."""
        stable = _stable_ts(400.0, 30)
        gc_drop = _shrinking_ts(400.0, 380.0, 5)
        regrowth = _growing_ts(380.0, 420.0, 15)
        ts = stable + gc_drop + regrowth
        phases = [Phase.STABLE, Phase.SHRINKING, Phase.GROWING]
        changepoints = [30, 35]
        segments = compute_segment_stats(ts, changepoints, phases)
        assert segments[1].label == SegmentLabel.GC_COLLECTION


# ---------------------------------------------------------------------------
# Full lifecycle with new labels
# ---------------------------------------------------------------------------


class TestNewLabelsLifecycle:
    def test_inference_server_lifecycle(self) -> None:
        """Init -> Active Inference -> Request Burst -> Active Inference."""
        import math
        init = _growing_ts(0.0, 2000.0, 10)
        # Active inference: stable with tiny variance
        infer = [2000.0 + 3.0 * math.sin(i * 0.3) for i in range(50)]
        burst = _volatile_ts(2100.0, 100.0, 10)
        # More inference
        infer2 = [2000.0 + 3.0 * math.sin(i * 0.3) for i in range(50)]
        ts = init + infer + burst + infer2
        phases = [
            Phase.GROWING, Phase.STABLE, Phase.VOLATILE, Phase.STABLE,
        ]
        changepoints = [10, 60, 70]
        segments = compute_segment_stats(ts, changepoints, phases)
        assert len(segments) == 4
        assert segments[0].label == SegmentLabel.INITIALIZATION
        assert segments[1].label == SegmentLabel.ACTIVE_INFERENCE
        assert segments[2].label == SegmentLabel.REQUEST_BURST
        assert segments[3].label == SegmentLabel.ACTIVE_INFERENCE

    def test_training_with_gc_and_epochs(self) -> None:
        """Init -> Batch -> Epoch Boundary -> Batch -> GC -> Batch."""
        init = _growing_ts(0.0, 800.0, 15)
        batch1 = _volatile_ts(800.0, 80.0, 30)
        epoch = _volatile_ts(780.0, 30.0, 8)
        batch2 = _volatile_ts(800.0, 80.0, 30)
        gc = _shrinking_ts(800.0, 770.0, 5)
        batch3 = _volatile_ts(800.0, 80.0, 30)
        ts = init + batch1 + epoch + batch2 + gc + batch3
        phases = [
            Phase.GROWING, Phase.VOLATILE, Phase.VOLATILE,
            Phase.VOLATILE, Phase.SHRINKING, Phase.VOLATILE,
        ]
        changepoints = [15, 45, 53, 83, 88]
        segments = compute_segment_stats(ts, changepoints, phases)
        assert len(segments) == 6
        assert segments[0].label == SegmentLabel.INITIALIZATION
        assert segments[1].label == SegmentLabel.BATCH_PROCESSING
        assert segments[2].label == SegmentLabel.EPOCH_BOUNDARY
        assert segments[3].label == SegmentLabel.BATCH_PROCESSING
        assert segments[4].label == SegmentLabel.GC_COLLECTION
        assert segments[5].label == SegmentLabel.BATCH_PROCESSING
