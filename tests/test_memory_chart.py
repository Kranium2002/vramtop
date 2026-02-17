"""Tests for memory chart rendering."""

from __future__ import annotations

import pytest
from rich.text import Text

from vramtop.analysis.phase_detector import Phase
from vramtop.analysis.segment_labels import (
    compute_segment_stats,
)
from vramtop.ui.widgets.memory_chart import (
    _downsample,
    _format_delta,
    _format_mb,
    _get_value_color,
    render_memory_chart,
    render_segment_summary,
)

# ---------------------------------------------------------------------------
# Downsample
# ---------------------------------------------------------------------------


class TestDownsample:
    def test_no_downsample_needed(self) -> None:
        ts = [1.0, 2.0, 3.0]
        assert _downsample(ts, 10) == [1.0, 2.0, 3.0]

    def test_exact_match(self) -> None:
        ts = [1.0, 2.0, 3.0]
        assert _downsample(ts, 3) == [1.0, 2.0, 3.0]

    def test_downsample_by_half(self) -> None:
        ts = [1.0, 3.0, 5.0, 7.0]
        result = _downsample(ts, 2)
        assert len(result) == 2
        assert result[0] == pytest.approx(2.0)
        assert result[1] == pytest.approx(6.0)

    def test_downsample_large(self) -> None:
        ts = list(range(100))
        result = _downsample(ts, 10)
        assert len(result) == 10

    def test_empty_input(self) -> None:
        assert _downsample([], 10) == []


# ---------------------------------------------------------------------------
# Color mapping
# ---------------------------------------------------------------------------


class TestColorMapping:
    def test_low_value_is_green(self) -> None:
        assert _get_value_color(10.0, 0.0, 100.0) == "#22c55e"

    def test_high_value_is_red(self) -> None:
        assert _get_value_color(95.0, 0.0, 100.0) == "#ef4444"

    def test_mid_value_is_yellow(self) -> None:
        assert _get_value_color(50.0, 0.0, 100.0) == "#facc15"

    def test_equal_min_max(self) -> None:
        assert _get_value_color(50.0, 50.0, 50.0) == "#22c55e"


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


class TestFormatting:
    def test_format_mb_small(self) -> None:
        assert _format_mb(512.0) == "512 MB"

    def test_format_mb_large(self) -> None:
        assert _format_mb(8192.0) == "8.0 GB"

    def test_format_delta_positive(self) -> None:
        assert _format_delta(500.0) == "+500 MB"

    def test_format_delta_negative_gb(self) -> None:
        assert _format_delta(-2048.0) == "-2.0 GB"

    def test_format_delta_no_change(self) -> None:
        assert _format_delta(0.5) == "no change"


# ---------------------------------------------------------------------------
# Chart rendering
# ---------------------------------------------------------------------------


class TestRenderChart:
    def test_renders_text_object(self) -> None:
        ts = [100.0, 200.0, 300.0, 400.0, 300.0, 200.0] * 5
        segments = compute_segment_stats(ts, [], [Phase.GROWING])
        result = render_memory_chart(ts, segments, width=44, height=8)
        assert isinstance(result, Text)
        assert len(result) > 0

    def test_insufficient_data(self) -> None:
        result = render_memory_chart([], [], width=44, height=8)
        assert isinstance(result, Text)
        assert "insufficient" in str(result).lower()

    def test_too_narrow(self) -> None:
        ts = [100.0] * 20
        segments = compute_segment_stats(ts, [], [Phase.STABLE])
        result = render_memory_chart(ts, segments, width=5, height=8)
        assert isinstance(result, Text)

    def test_chart_contains_sparkline(self) -> None:
        ts = [100.0, 200.0, 300.0, 400.0, 500.0] * 6
        segments = compute_segment_stats(ts, [], [Phase.GROWING])
        result = render_memory_chart(ts, segments, width=44, height=8)
        text = result.plain
        # Should contain sparkline block chars
        assert any(c in text for c in "▁▂▃▄▅▆▇█")

    def test_chart_shows_range(self) -> None:
        ts = [100.0, 500.0] * 10
        segments = compute_segment_stats(ts, [], [Phase.VOLATILE])
        result = render_memory_chart(ts, segments, width=44, height=8)
        text = result.plain
        assert "MB" in text or "GB" in text

    def test_chart_with_changepoints(self) -> None:
        growing = [float(i * 10) for i in range(20)]
        stable = [190.0] * 20
        ts = growing + stable
        segments = compute_segment_stats(
            ts, [20], [Phase.GROWING, Phase.STABLE]
        )
        result = render_memory_chart(ts, segments, width=44, height=8)
        assert isinstance(result, Text)
        text = result.plain
        # Sparkline should be present
        assert any(c in text for c in "▁▂▃▄▅▆▇█")
        # Range should show min and peak
        assert "MB" in text
        assert "peak" in text

    def test_chart_compact(self) -> None:
        """Chart should be compact (3 rows max: range + sparkline + bar)."""
        ts = [float(i * 10) for i in range(40)]
        segments = compute_segment_stats(ts, [], [Phase.GROWING])
        result = render_memory_chart(ts, segments, width=44, height=8)
        lines = result.plain.strip().split("\n")
        assert len(lines) <= 3


# ---------------------------------------------------------------------------
# Segment summary
# ---------------------------------------------------------------------------


class TestSegmentSummary:
    def test_empty_segments(self) -> None:
        result = render_segment_summary([])
        assert "no segments" in str(result).lower()

    def test_single_segment(self) -> None:
        ts = [100.0] * 30
        segments = compute_segment_stats(ts, [], [Phase.STABLE])
        result = render_segment_summary(segments)
        text = str(result)
        assert "Steady State" in text

    def test_multiple_segments(self) -> None:
        growing = [float(i * 20) for i in range(20)]
        stable = [380.0] * 20
        ts = growing + stable
        segments = compute_segment_stats(
            ts, [20], [Phase.GROWING, Phase.STABLE]
        )
        result = render_segment_summary(segments)
        text = str(result)
        # Both segments should appear as one-line entries
        assert "Initialization" in text
        assert "Steady State" in text

    def test_shows_delta(self) -> None:
        ts = [float(i * 10) for i in range(20)]
        segments = compute_segment_stats(ts, [], [Phase.GROWING])
        result = render_segment_summary(segments)
        text = str(result)
        assert "+" in text or "MB" in text

    def test_shows_mean_level(self) -> None:
        ts = [100.0] * 30
        segments = compute_segment_stats(ts, [], [Phase.STABLE])
        result = render_segment_summary(segments)
        text = str(result)
        assert "100 MB" in text

    def test_shows_gpu_percentage(self) -> None:
        ts = [8000.0] * 30
        segments = compute_segment_stats(
            ts, [], [Phase.STABLE], gpu_total_mb=16000.0,
        )
        result = render_segment_summary(segments, gpu_total_mb=16000.0)
        text = str(result)
        assert "50%" in text

    def test_shows_description(self) -> None:
        ts = [100.0] * 30
        segments = compute_segment_stats(ts, [], [Phase.STABLE])
        result = render_segment_summary(segments)
        text = str(result)
        assert "Stable memory usage" in text

    def test_large_values_show_gb(self) -> None:
        ts = [8000.0] * 30
        segments = compute_segment_stats(ts, [], [Phase.STABLE])
        result = render_segment_summary(segments)
        text = str(result)
        assert "GB" in text

    def test_shows_duration(self) -> None:
        ts = [100.0] * 30
        segments = compute_segment_stats(ts, [], [Phase.STABLE])
        result = render_segment_summary(segments)
        text = str(result)
        assert "30s" in text
