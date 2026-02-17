"""VRAM memory timeline with sparkline and segment analysis.

Renders a Rich-based visualization inside the detail panel showing:
- A single-row sparkline of memory usage over time
- A compact one-line-per-segment timeline below it
"""

from __future__ import annotations

from rich.text import Text

from vramtop.analysis.segment_labels import (
    LABEL_COLORS,
    LABEL_DESCRIPTIONS,
    SegmentInfo,
    SegmentLabel,
)

# Sparkline characters (bottom to top, 8 levels)
_SPARK = "▁▂▃▄▅▆▇█"

# Direction indicators per phase pattern
_ARROWS: dict[SegmentLabel, str] = {
    # Growing
    SegmentLabel.INITIALIZATION: "▲",
    SegmentLabel.PRE_ALLOCATION: "▲",
    SegmentLabel.WARMUP: "▲",
    SegmentLabel.ALLOCATION_EVENT: "▲",
    SegmentLabel.MEMORY_GROWTH: "▲",
    SegmentLabel.MEMORY_LEAK: "!",
    SegmentLabel.CACHE_FILLING: "▲",
    SegmentLabel.GRADIENT_STEPS: "▲",
    SegmentLabel.OPTIMIZER_INIT: "▲",
    SegmentLabel.MEMORY_PRESSURE: "!",
    # Stable
    SegmentLabel.STEADY_STATE: "─",
    SegmentLabel.SATURATION: "!",
    SegmentLabel.PLATEAU: "─",
    SegmentLabel.IDLE: "·",
    SegmentLabel.ACTIVE_INFERENCE: "─",
    # Volatile
    SegmentLabel.BATCH_PROCESSING: "~",
    SegmentLabel.FRAGMENTATION: "~",
    SegmentLabel.CHECKPOINT_SAVE: "~",
    SegmentLabel.EPOCH_BOUNDARY: "~",
    SegmentLabel.REQUEST_BURST: "~",
    # Shrinking
    SegmentLabel.CLEANUP: "▼",
    SegmentLabel.RELEASING: "▼",
    SegmentLabel.COOLDOWN: "▼",
    SegmentLabel.CACHE_EVICTION: "▼",
    SegmentLabel.GC_COLLECTION: "▼",
}


def render_memory_chart(
    timeseries: list[float],
    segments: list[SegmentInfo],
    width: int = 44,
    height: int = 8,
) -> Text:
    """Render a sparkline with min/max range.

    Parameters
    ----------
    timeseries:
        Memory values in MB.
    segments:
        Labeled segment info (unused in chart, used in summary).
    width:
        Available width in characters.
    height:
        Ignored (kept for API compatibility).

    Returns
    -------
    A Rich Text renderable with the sparkline.
    """
    if not timeseries or width < 10:
        return Text("(insufficient data)")

    chart_width = min(width - 2, len(timeseries))
    if chart_width < 5:
        return Text("(too narrow)")

    # Downsample timeseries to fit
    ts = _downsample(timeseries, chart_width)
    n = len(ts)

    min_val = min(ts)
    max_val = max(ts)
    if max_val <= min_val:
        max_val = min_val + 1.0

    result = Text()

    # Row 1: Sparkline
    result.append("  ")
    for x in range(n):
        val = ts[x]
        frac = (val - min_val) / (max_val - min_val)
        frac = max(0.0, min(1.0, frac))
        idx = int(frac * (len(_SPARK) - 1))
        color = _get_value_color(val, min_val, max_val)
        result.append(_SPARK[idx], style=color)
    result.append("\n")

    # Row 2: Range (min left, max right)
    min_str = _format_mb(min_val)
    max_str = f"{_format_mb(max_val)} peak"
    gap = width - len(f"  {min_str}") - len(max_str)
    result.append(f"  {min_str}", style="dim")
    if gap > 0:
        result.append(" " * gap, style="dim")
    result.append(max_str, style="dim")

    return result


def render_segment_summary(
    segments: list[SegmentInfo],
    gpu_total_mb: float = 0.0,
) -> Text:
    """Render a compact one-line-per-segment timeline.

    Each line: arrow + label + delta/level + duration
    """
    if not segments:
        return Text("(no segments)")

    result = Text()

    # Find max label width for alignment
    max_label_len = max(len(seg.label.value) for seg in segments)

    for i, seg in enumerate(segments):
        if i > 0:
            result.append("\n")

        color = LABEL_COLORS.get(seg.label, "white")
        arrow = _ARROWS.get(seg.label, " ")

        # Arrow (colored)
        result.append(f"  {arrow} ", style=color)

        # Label (colored, padded for alignment)
        label_str = seg.label.value
        result.append(label_str, style=color)
        pad = max_label_len - len(label_str) + 2
        result.append(" " * pad)

        # Value column: show the most useful stat per type
        val_str = _segment_value(seg, gpu_total_mb)
        result.append(val_str, style="dim")

        # Duration (right-aligned feel)
        dur_str = f"  {seg.duration_samples}s"
        result.append(dur_str, style="dim")

    # After listing, show description for current (last) segment
    if segments:
        current = segments[-1]
        desc = LABEL_DESCRIPTIONS.get(current.label, "")
        if desc:
            result.append("\n\n")
            color = LABEL_COLORS.get(current.label, "white")
            result.append("  Now: ", style="bold")
            result.append(current.label.value, style=color)
            result.append(f"\n  {desc}", style="dim")

    return result


def _segment_value(seg: SegmentInfo, gpu_total_mb: float) -> str:
    """Format the most useful value for a segment."""
    from vramtop.analysis.phase_detector import Phase

    if seg.phase == Phase.STABLE:
        level = _format_mb(seg.mean_mb)
        if gpu_total_mb > 0:
            pct = seg.mean_mb / gpu_total_mb * 100
            return f"{level} ({pct:.0f}%)"
        return level

    if seg.phase == Phase.VOLATILE:
        # Show amplitude for volatile segments
        if seg.variance > 0:
            import math

            amp = math.sqrt(seg.variance) * 2  # rough peak-to-peak
            if amp >= 1.0:
                return f"\u00b1{_format_mb(amp)}"
        return _format_mb(seg.mean_mb)

    # Growing or shrinking: show delta
    return _format_delta(seg.delta_mb)


def _format_mb(mb: float) -> str:
    """Format MB value with appropriate unit."""
    if mb >= 1024:
        return f"{mb / 1024:.1f} GB"
    return f"{mb:.0f} MB"


def _format_delta(delta_mb: float) -> str:
    """Format a memory delta with sign and appropriate unit."""
    if abs(delta_mb) < 1.0:
        return "no change"
    sign = "+" if delta_mb > 0 else ""
    if abs(delta_mb) >= 1024:
        return f"{sign}{delta_mb / 1024:.1f} GB"
    return f"{sign}{delta_mb:.0f} MB"


def _downsample(timeseries: list[float], target_len: int) -> list[float]:
    """Downsample a timeseries to target_len points using averaging."""
    n = len(timeseries)
    if n <= target_len:
        return list(timeseries)

    result: list[float] = []
    bucket_size = n / target_len
    for i in range(target_len):
        start = int(i * bucket_size)
        end = int((i + 1) * bucket_size)
        end = min(end, n)
        if start < end:
            bucket = timeseries[start:end]
            result.append(sum(bucket) / len(bucket))
        elif result:
            result.append(result[-1])
    return result


def _get_value_color(val: float, min_val: float, max_val: float) -> str:
    """Get a color based on value position in range (green -> yellow -> red)."""
    if max_val <= min_val:
        return "#22c55e"
    ratio = (val - min_val) / (max_val - min_val)
    ratio = max(0.0, min(1.0, ratio))
    if ratio < 0.4:
        return "#22c55e"  # green
    if ratio < 0.7:
        return "#facc15"  # yellow
    if ratio < 0.9:
        return "#fb923c"  # orange
    return "#ef4444"  # red
