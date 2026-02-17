"""Model-agnostic heuristic labels for PELT memory segments.

Given a timeseries, changepoints, and phase classifications, assigns
human-readable labels like "Initialization", "Steady State", or
"Batch Processing" based purely on memory access patterns.

The labeling is deliberately framework- and model-agnostic — it works
for PyTorch training, vLLM inference, Ollama, or any GPU process.

Labels are assigned in two passes:
1. Single-segment heuristics (phase, position, magnitude, duration, variance)
2. Multi-segment refinement (neighboring context for checkpoint saves, cooldowns, etc.)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from vramtop.analysis.phase_detector import Phase


class SegmentLabel(Enum):
    """Human-readable labels for memory segments."""

    # --- Growth patterns ---
    INITIALIZATION = "Initialization"
    PRE_ALLOCATION = "Pre-allocation"
    WARMUP = "Warmup"
    ALLOCATION_EVENT = "Allocation Event"
    MEMORY_GROWTH = "Memory Growth"
    MEMORY_LEAK = "Memory Leak"
    CACHE_FILLING = "Cache Filling"
    GRADIENT_STEPS = "Gradient Steps"
    OPTIMIZER_INIT = "Optimizer Init"
    MEMORY_PRESSURE = "Memory Pressure"

    # --- Stable patterns ---
    STEADY_STATE = "Steady State"
    SATURATION = "Saturation"
    PLATEAU = "Plateau"
    IDLE = "Idle"
    ACTIVE_INFERENCE = "Active Inference"

    # --- Volatile patterns ---
    BATCH_PROCESSING = "Batch Processing"
    FRAGMENTATION = "Fragmentation"
    CHECKPOINT_SAVE = "Checkpoint Save"
    EPOCH_BOUNDARY = "Epoch Boundary"
    REQUEST_BURST = "Request Burst"

    # --- Shrinking patterns ---
    CLEANUP = "Cleanup"
    RELEASING = "Releasing Memory"
    COOLDOWN = "Cooldown"
    CACHE_EVICTION = "Cache Eviction"
    GC_COLLECTION = "GC Collection"


# Color mapping for segment labels (Rich markup)
LABEL_COLORS: dict[SegmentLabel, str] = {
    SegmentLabel.INITIALIZATION: "bold cyan",
    SegmentLabel.PRE_ALLOCATION: "bold bright_cyan",
    SegmentLabel.WARMUP: "yellow",
    SegmentLabel.ALLOCATION_EVENT: "bold red",
    SegmentLabel.MEMORY_GROWTH: "bold yellow",
    SegmentLabel.MEMORY_LEAK: "bold red on dark_red",
    SegmentLabel.CACHE_FILLING: "bright_yellow",
    SegmentLabel.GRADIENT_STEPS: "bright_magenta",
    SegmentLabel.OPTIMIZER_INIT: "bold yellow",
    SegmentLabel.MEMORY_PRESSURE: "bold red",
    SegmentLabel.STEADY_STATE: "green",
    SegmentLabel.SATURATION: "bold bright_red",
    SegmentLabel.PLATEAU: "dim green",
    SegmentLabel.IDLE: "dim",
    SegmentLabel.ACTIVE_INFERENCE: "bright_green",
    SegmentLabel.BATCH_PROCESSING: "magenta",
    SegmentLabel.FRAGMENTATION: "bright_magenta",
    SegmentLabel.CHECKPOINT_SAVE: "bright_cyan",
    SegmentLabel.EPOCH_BOUNDARY: "bright_yellow",
    SegmentLabel.REQUEST_BURST: "bold magenta",
    SegmentLabel.CLEANUP: "cyan",
    SegmentLabel.RELEASING: "blue",
    SegmentLabel.COOLDOWN: "bright_blue",
    SegmentLabel.CACHE_EVICTION: "bright_cyan",
    SegmentLabel.GC_COLLECTION: "dim cyan",
}

# Short labels for constrained UI space (max 7 chars)
LABEL_SHORT: dict[SegmentLabel, str] = {
    SegmentLabel.INITIALIZATION: "Init",
    SegmentLabel.PRE_ALLOCATION: "PreAll",
    SegmentLabel.WARMUP: "Warm",
    SegmentLabel.ALLOCATION_EVENT: "Alloc",
    SegmentLabel.MEMORY_GROWTH: "Growth",
    SegmentLabel.MEMORY_LEAK: "Leak!",
    SegmentLabel.CACHE_FILLING: "Cache",
    SegmentLabel.GRADIENT_STEPS: "GradSt",
    SegmentLabel.OPTIMIZER_INIT: "OptInit",
    SegmentLabel.MEMORY_PRESSURE: "Press!",
    SegmentLabel.STEADY_STATE: "Steady",
    SegmentLabel.SATURATION: "Satur!",
    SegmentLabel.PLATEAU: "Flat",
    SegmentLabel.IDLE: "Idle",
    SegmentLabel.ACTIVE_INFERENCE: "Infer",
    SegmentLabel.BATCH_PROCESSING: "Batch",
    SegmentLabel.FRAGMENTATION: "Frag",
    SegmentLabel.CHECKPOINT_SAVE: "Ckpt",
    SegmentLabel.EPOCH_BOUNDARY: "Epoch",
    SegmentLabel.REQUEST_BURST: "Burst",
    SegmentLabel.CLEANUP: "Clean",
    SegmentLabel.RELEASING: "Release",
    SegmentLabel.COOLDOWN: "Cool",
    SegmentLabel.CACHE_EVICTION: "Evict",
    SegmentLabel.GC_COLLECTION: "GC",
}

# Direction arrows for chart axis labels
LABEL_ARROW: dict[SegmentLabel, str] = {
    SegmentLabel.INITIALIZATION: "^^",
    SegmentLabel.PRE_ALLOCATION: "^^",
    SegmentLabel.WARMUP: "^",
    SegmentLabel.ALLOCATION_EVENT: "^",
    SegmentLabel.MEMORY_GROWTH: "^",
    SegmentLabel.MEMORY_LEAK: "^!",
    SegmentLabel.CACHE_FILLING: "^",
    SegmentLabel.GRADIENT_STEPS: "^~",
    SegmentLabel.OPTIMIZER_INIT: "^",
    SegmentLabel.MEMORY_PRESSURE: "^!",
    SegmentLabel.STEADY_STATE: "--",
    SegmentLabel.SATURATION: "!!",
    SegmentLabel.PLATEAU: "--",
    SegmentLabel.IDLE: "..",
    SegmentLabel.ACTIVE_INFERENCE: "~-",
    SegmentLabel.BATCH_PROCESSING: "~~",
    SegmentLabel.FRAGMENTATION: "~~",
    SegmentLabel.CHECKPOINT_SAVE: "~",
    SegmentLabel.EPOCH_BOUNDARY: "v~",
    SegmentLabel.REQUEST_BURST: "~^",
    SegmentLabel.CLEANUP: "vv",
    SegmentLabel.RELEASING: "v",
    SegmentLabel.COOLDOWN: "v",
    SegmentLabel.CACHE_EVICTION: "v",
    SegmentLabel.GC_COLLECTION: "v~",
}

# Human-readable descriptions of what each label means
LABEL_DESCRIPTIONS: dict[SegmentLabel, str] = {
    SegmentLabel.INITIALIZATION: "Loading model weights onto GPU",
    SegmentLabel.PRE_ALLOCATION: "Framework reserving GPU memory pool",
    SegmentLabel.WARMUP: "Allocating optimizer states and buffers",
    SegmentLabel.ALLOCATION_EVENT: "Single large allocation burst",
    SegmentLabel.MEMORY_GROWTH: "Gradual increase in memory usage",
    SegmentLabel.MEMORY_LEAK: "Unbounded growth, possible memory leak",
    SegmentLabel.CACHE_FILLING: "KV cache or buffer pool expanding",
    SegmentLabel.GRADIENT_STEPS: "Step-wise growth from gradient accumulation",
    SegmentLabel.OPTIMIZER_INIT: "Allocating optimizer states (2-3x model size)",
    SegmentLabel.MEMORY_PRESSURE: "Growing near GPU limit, OOM risk increasing",
    SegmentLabel.STEADY_STATE: "Stable memory usage, normal operation",
    SegmentLabel.SATURATION: "Near GPU memory limit, risk of OOM",
    SegmentLabel.PLATEAU: "Brief stable pause between transitions",
    SegmentLabel.IDLE: "Process idle, minimal GPU memory used",
    SegmentLabel.ACTIVE_INFERENCE: "Serving requests with small memory fluctuations",
    SegmentLabel.BATCH_PROCESSING: "Fluctuating memory from batch workload",
    SegmentLabel.FRAGMENTATION: "Small oscillations, possible fragmentation",
    SegmentLabel.CHECKPOINT_SAVE: "Temporary spike from saving checkpoint",
    SegmentLabel.EPOCH_BOUNDARY: "Brief memory dip between training epochs",
    SegmentLabel.REQUEST_BURST: "Short spike from concurrent request surge",
    SegmentLabel.CLEANUP: "Rapidly releasing GPU memory",
    SegmentLabel.RELEASING: "Gradually freeing memory",
    SegmentLabel.COOLDOWN: "Winding down after active processing",
    SegmentLabel.CACHE_EVICTION: "Evicting cached data to free memory",
    SegmentLabel.GC_COLLECTION: "Brief release from garbage collection cycle",
}


@dataclass(frozen=True)
class SegmentInfo:
    """Rich segment metadata for display."""

    label: SegmentLabel
    phase: Phase
    start_idx: int
    end_idx: int
    mean_mb: float
    delta_mb: float  # total change: end - start
    rate_mb_per_sample: float
    duration_samples: int
    variance: float


def compute_segment_stats(
    timeseries: list[float],
    changepoints: list[int],
    phases: list[Phase],
    gpu_total_mb: float = 0.0,
) -> list[SegmentInfo]:
    """Compute statistics for each segment and assign heuristic labels.

    Parameters
    ----------
    timeseries:
        Memory values in MB.
    changepoints:
        Indices where segments change.
    phases:
        Phase classification per segment (from PELTDetector.classify_segments).
    gpu_total_mb:
        Total GPU memory in MB (used for saturation detection). 0 = unknown.

    Returns
    -------
    List of SegmentInfo with heuristic labels, one per segment.
    """
    if not timeseries or not phases:
        return []

    boundaries = [0, *changepoints, len(timeseries)]
    total_len = len(timeseries)

    # First pass: compute stats and assign initial labels
    raw_segments: list[_RawSegment] = []
    for i, phase in enumerate(phases):
        start = boundaries[i]
        end = boundaries[i + 1]
        segment = timeseries[start:end]

        if not segment:
            continue

        mean_mb = sum(segment) / len(segment)
        delta_mb = segment[-1] - segment[0]
        duration = len(segment)
        rate = delta_mb / max(duration, 1)
        min_val = min(segment)
        max_val = max(segment)

        # Variance of deltas within segment
        deltas: list[float] = []
        if len(segment) >= 2:
            deltas = [segment[j + 1] - segment[j] for j in range(len(segment) - 1)]
            mean_delta = sum(deltas) / len(deltas)
            variance = sum((d - mean_delta) ** 2 for d in deltas) / len(deltas)
        else:
            variance = 0.0

        position_frac = start / max(total_len, 1)

        label = _assign_label(
            phase=phase,
            position_frac=position_frac,
            delta_mb=delta_mb,
            abs_delta_mb=abs(delta_mb),
            rate=rate,
            duration=duration,
            variance=variance,
            mean_mb=mean_mb,
            min_val=min_val,
            max_val=max_val,
            deltas=deltas,
            gpu_total_mb=gpu_total_mb,
        )

        raw_segments.append(
            _RawSegment(
                label=label,
                phase=phase,
                start_idx=start,
                end_idx=end,
                mean_mb=mean_mb,
                delta_mb=delta_mb,
                rate=rate,
                duration=duration,
                variance=variance,
                min_val=min_val,
                max_val=max_val,
            )
        )

    # Second pass: refine labels using multi-segment context
    _refine_labels(raw_segments)

    return [
        SegmentInfo(
            label=seg.label,
            phase=seg.phase,
            start_idx=seg.start_idx,
            end_idx=seg.end_idx,
            mean_mb=seg.mean_mb,
            delta_mb=seg.delta_mb,
            rate_mb_per_sample=seg.rate,
            duration_samples=seg.duration,
            variance=seg.variance,
        )
        for seg in raw_segments
    ]


@dataclass
class _RawSegment:
    """Mutable intermediate segment used during two-pass labeling."""

    label: SegmentLabel
    phase: Phase
    start_idx: int
    end_idx: int
    mean_mb: float
    delta_mb: float
    rate: float
    duration: int
    variance: float
    min_val: float
    max_val: float


def _assign_label(
    *,
    phase: Phase,
    position_frac: float,
    delta_mb: float,
    abs_delta_mb: float,
    rate: float,
    duration: int,
    variance: float,
    mean_mb: float,
    min_val: float,
    max_val: float,
    deltas: list[float],
    gpu_total_mb: float,
) -> SegmentLabel:
    """Assign a human-readable label based on memory pattern heuristics.

    Heuristics are ordered from most specific to most general.
    """
    is_early = position_frac < 0.25
    is_very_early = position_frac < 0.05
    is_late = position_frac > 0.60
    is_large_change = abs_delta_mb > 50.0
    is_very_large_change = abs_delta_mb > 200.0
    is_short = duration < 8
    is_long = duration > 40
    is_high_variance = variance > 100.0

    # ── VOLATILE ──────────────────────────────────────────────────
    # Epoch boundary and request burst are detected in Pass 2 (need context)

    if phase == Phase.VOLATILE:
        # Fragmentation: oscillating around same level, low net change
        if abs_delta_mb < 20.0 and not is_high_variance:
            return SegmentLabel.FRAGMENTATION
        # Very short volatile spike
        if is_short:
            return SegmentLabel.CHECKPOINT_SAVE
        return SegmentLabel.BATCH_PROCESSING

    # ── GROWING ───────────────────────────────────────────────────

    if phase == Phase.GROWING:
        # Memory pressure: growing when already at >70% GPU
        if (
            gpu_total_mb > 0
            and mean_mb > gpu_total_mb * 0.70
            and not is_short
        ):
            return SegmentLabel.MEMORY_PRESSURE
        # Pre-allocation: massive jump at very start, very short
        if is_very_early and is_very_large_change and is_short:
            return SegmentLabel.PRE_ALLOCATION
        # Large rapid increase at start = initialization (model loading)
        if is_early and is_very_large_change:
            return SegmentLabel.INITIALIZATION
        # Moderate increase at start = warmup (optimizer, buffers)
        if is_early and is_large_change:
            return SegmentLabel.WARMUP
        # Gradient steps: staircase pattern (low variance in deltas, but
        # clear upward trend with periodic flat segments).
        # Only after early startup phases — pre-allocation and model
        # loading also produce step-wise patterns but are NOT gradient
        # accumulation.
        if not is_early and len(deltas) >= 5 and _is_staircase(deltas):
            return SegmentLabel.GRADIENT_STEPS
        # Short duration burst = single allocation event
        if is_short and is_large_change:
            return SegmentLabel.ALLOCATION_EVENT
        # Memory leak: late, long, sustained growth
        if is_late and is_long:
            return SegmentLabel.MEMORY_LEAK
        # Cache filling: moderate growth after the start phase
        if not is_early and not is_late and is_long:
            return SegmentLabel.CACHE_FILLING
        # Early moderate growth
        if is_early:
            return SegmentLabel.WARMUP
        # Late gradual growth
        return SegmentLabel.MEMORY_GROWTH

    # ── SHRINKING ─────────────────────────────────────────────────

    # Cache eviction is detected in Pass 2 (needs neighboring context)
    if phase == Phase.SHRINKING:
        # GC collection: very brief drop with small delta
        if is_short and abs_delta_mb < 50.0:
            return SegmentLabel.GC_COLLECTION
        if is_very_large_change:
            return SegmentLabel.CLEANUP
        # Cooldown: gradual slow decrease (will be refined by multi-segment
        # pass if preceded by volatile/batch)
        if is_long and not is_large_change:
            return SegmentLabel.COOLDOWN
        return SegmentLabel.RELEASING

    # ── STABLE ────────────────────────────────────────────────────

    if mean_mb < 10.0:
        return SegmentLabel.IDLE
    if duration < 5:
        return SegmentLabel.PLATEAU
    # Saturation: stable at very high GPU utilization
    if gpu_total_mb > 0 and mean_mb > gpu_total_mb * 0.85:
        return SegmentLabel.SATURATION
    # Active inference: stable with very small fluctuations (variance > 0)
    if variance > 0.0 and variance < 50.0 and is_long:
        return SegmentLabel.ACTIVE_INFERENCE
    return SegmentLabel.STEADY_STATE


def _is_staircase(deltas: list[float]) -> bool:
    """Detect staircase (gradient accumulation) pattern in deltas.

    A staircase has alternating runs of near-zero deltas (flat steps)
    and positive jumps. We detect this by counting zero-crossings
    relative to the mean and checking that a significant fraction
    of deltas are near zero while some are significantly positive.

    Requires at least 3 distinct jumps to distinguish from
    pre-allocation / KV cache setup which may have only 1-2 steps.
    """
    if not deltas:
        return False

    mean_d = sum(deltas) / len(deltas)
    if mean_d <= 0:
        return False

    threshold = mean_d * 0.3
    near_zero = sum(1 for d in deltas if abs(d) < threshold)
    large_positive = sum(1 for d in deltas if d > mean_d * 1.5)

    # Staircase: many flat samples + multiple big jumps.
    # Require >= 3 actual jumps to avoid false positives from
    # framework pre-allocation (1-2 chunk allocations).
    frac_flat = near_zero / len(deltas)

    return frac_flat > 0.4 and large_positive >= 3


def _refine_labels(segments: list[_RawSegment]) -> None:
    """Second pass: refine labels using multi-segment context.

    Looks at neighboring segments to detect patterns that require
    broader context:
    - Volatile segment between two stable segments at same level → Checkpoint Save
    - Shrinking segment after volatile → Cooldown
    - Growing segment between two stable segments → Cache Filling (if level rises)
    """
    n = len(segments)
    for i in range(n):
        seg = segments[i]
        prev_seg = segments[i - 1] if i > 0 else None
        next_seg = segments[i + 1] if i < n - 1 else None

        # Request burst: short volatile after steady/inference
        # But NOT when both sides are stable at same level (that's checkpoint)
        # Exception: Active Inference context = request burst, not checkpoint
        _is_ckpt_pattern = (
            next_seg is not None
            and next_seg.phase == Phase.STABLE
            and prev_seg is not None
            and prev_seg.phase == Phase.STABLE
            and prev_seg.label != SegmentLabel.ACTIVE_INFERENCE
            and abs(prev_seg.mean_mb - next_seg.mean_mb) < 50.0
        )
        if (
            seg.phase == Phase.VOLATILE
            and seg.duration < 12
            and not _is_ckpt_pattern
            and prev_seg is not None
            and prev_seg.label in (
                SegmentLabel.STEADY_STATE,
                SegmentLabel.ACTIVE_INFERENCE,
            )
        ):
            seg.label = SegmentLabel.REQUEST_BURST

        # Volatile between two stable segments at ~same level → Checkpoint Save
        # (but not when surrounding is Active Inference — that's a request burst)
        _surround_is_inference = (
            prev_seg is not None
            and next_seg is not None
            and prev_seg.label == SegmentLabel.ACTIVE_INFERENCE
            and next_seg.label == SegmentLabel.ACTIVE_INFERENCE
        )
        if (
            seg.phase == Phase.VOLATILE
            and seg.label != SegmentLabel.REQUEST_BURST
            and not _surround_is_inference
            and seg.duration < 15
            and prev_seg is not None
            and next_seg is not None
            and prev_seg.phase == Phase.STABLE
            and next_seg.phase == Phase.STABLE
            and abs(prev_seg.mean_mb - next_seg.mean_mb) < 50.0
        ):
            seg.label = SegmentLabel.CHECKPOINT_SAVE

        # Shrinking after volatile/batch → Cooldown (not GC though)
        if (
            seg.phase == Phase.SHRINKING
            and seg.label == SegmentLabel.RELEASING
            and prev_seg is not None
            and prev_seg.phase == Phase.VOLATILE
        ):
            seg.label = SegmentLabel.COOLDOWN

        # Growing between stable segments where level rises → Cache Filling
        if (
            seg.phase == Phase.GROWING
            and seg.label == SegmentLabel.MEMORY_GROWTH
            and next_seg is not None
            and next_seg.phase == Phase.STABLE
            and seg.delta_mb > 20.0
        ):
            seg.label = SegmentLabel.CACHE_FILLING

        # Optimizer init: growth right after initialization segment
        # (second growing phase following the primary model load)
        # Don't override more specific labels like gradient steps
        if (
            seg.phase == Phase.GROWING
            and seg.label not in (
                SegmentLabel.GRADIENT_STEPS,
                SegmentLabel.MEMORY_PRESSURE,
                SegmentLabel.PRE_ALLOCATION,
            )
            and prev_seg is not None
            and prev_seg.label == SegmentLabel.INITIALIZATION
        ):
            seg.label = SegmentLabel.OPTIMIZER_INIT

        # Epoch boundary: short volatile between two batch segments
        if (
            seg.phase == Phase.VOLATILE
            and seg.label == SegmentLabel.BATCH_PROCESSING
            and seg.duration < 15
            and prev_seg is not None
            and next_seg is not None
            and prev_seg.label == SegmentLabel.BATCH_PROCESSING
        ):
            seg.label = SegmentLabel.EPOCH_BOUNDARY

        # Cache eviction: moderate shrink after stable/inference (not batch)
        if (
            seg.phase == Phase.SHRINKING
            and seg.label == SegmentLabel.RELEASING
            and prev_seg is not None
            and prev_seg.label in (
                SegmentLabel.STEADY_STATE,
                SegmentLabel.ACTIVE_INFERENCE,
                SegmentLabel.CACHE_FILLING,
            )
            and not seg.duration > 40
        ):
            seg.label = SegmentLabel.CACHE_EVICTION
