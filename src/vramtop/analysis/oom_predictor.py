"""Range-based, confidence-gated OOM prediction.

Key design principles:
- Use GPU-level memory trends (not per-process) since OOM is a GPU-wide event.
- Require BOTH sustained growth AND high utilization before alerting.
- Severity levels: warning (yellow) vs critical (red) based on time remaining.
- Always show ranges, never point estimates.
- Err on the side of silence — a false alarm trains users to ignore alerts.
- Use sliding window (not EMA) for growth detection — reacts quickly to stops.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum


class Severity(Enum):
    NONE = "none"
    WARNING = "warning"  # >50% used and growing
    CRITICAL = "critical"  # <5 min to OOM at current rate


@dataclass
class OOMPrediction:
    seconds_low: float | None  # Optimistic (fastest growth in phase)
    seconds_high: float | None  # Pessimistic (slowest growth in phase)
    confidence: float  # 0.0 - 1.0
    severity: Severity
    display: str  # "OOM in ~60-120s" or "Stable" or "Low VRAM (85%)"
    rate_range_mb_per_sec: tuple[float, float] | None = None
    utilization_pct: float = 0.0


# Thresholds
_MIN_CONFIDENCE = 0.3  # Don't show alerts below this
_MIN_UTILIZATION_PCT = 50.0  # Don't predict OOM if <50% used
_HIGH_UTILIZATION_PCT = 85.0  # Show "Low VRAM" even without growth
_CRITICAL_SECONDS = 300.0  # <5 min = critical severity
_CAP_SECONDS = 3600.0  # Don't show predictions beyond 1 hour
_WINDOW_SIZE = 5  # Sliding window for growth detection
_GROWTH_FRACTION = 0.6  # >=60% of window samples must be positive


class OOMPredictor:
    """GPU-level range-based OOM predictor.

    Uses a sliding window of recent memory deltas for growth detection.
    This reacts quickly when growth stops (unlike EMA which has a long tail).

    Rules:
    1. Only predict if growth sustained > min_sustained_samples.
    2. Only predict if utilization > 50% (plenty of room = no concern).
    3. Require confidence > 0.3 before displaying alerts.
    4. Use rate range (min/max over window) for confidence interval.
    5. Never show point estimate — always a range.
    6. Cap at ">1h". Suppress noise below min_rate threshold.
    7. Show "Low VRAM" warning at >85% even without active growth.
    """

    def __init__(
        self,
        min_sustained_samples: int = 10,
        min_rate_mb_per_sec: float = 5.0,
    ) -> None:
        self.min_sustained_samples = min_sustained_samples
        self.min_rate_mb_per_sec = min_rate_mb_per_sec

        self._window: deque[float] = deque(maxlen=_WINDOW_SIZE)
        self._growing_samples: int = 0
        self._growing_min_rate: float | None = None
        self._growing_max_rate: float | None = None

    def update(
        self,
        used_mb: float,
        free_mb: float,
        total_mb: float,
        delta_mb: float,
        dt_seconds: float,
    ) -> OOMPrediction:
        """Update prediction with GPU-level memory data.

        Args:
            used_mb: Current application-level used memory (MB).
            free_mb: Truly allocatable free memory (MB).
            total_mb: Total GPU memory (MB).
            delta_mb: Change in used_mb since last sample.
            dt_seconds: Time since last sample.
        """
        utilization = (used_mb / max(total_mb, 1.0)) * 100.0
        rate = delta_mb / max(dt_seconds, 0.001)

        # Add to sliding window
        self._window.append(rate)

        # Determine if GPU memory is actively growing using the window.
        # Require >= 60% of recent samples to show positive growth above noise.
        if len(self._window) >= 3:
            positive_count = sum(
                1 for r in self._window if r > self.min_rate_mb_per_sec
            )
            positive_frac = positive_count / len(self._window)
            median_rate = sorted(self._window)[len(self._window) // 2]
            is_growing = (
                positive_frac >= _GROWTH_FRACTION
                and median_rate > self.min_rate_mb_per_sec
            )
        else:
            is_growing = False
            median_rate = 0.0

        if is_growing:
            self._growing_samples += 1
            # Track min/max of median rate for range computation
            if self._growing_min_rate is None or median_rate < self._growing_min_rate:
                self._growing_min_rate = median_rate
            if self._growing_max_rate is None or median_rate > self._growing_max_rate:
                self._growing_max_rate = median_rate
        else:
            self._growing_samples = 0
            self._growing_min_rate = None
            self._growing_max_rate = None

        # Rule 7: High utilization warning (even without growth)
        if utilization >= _HIGH_UTILIZATION_PCT and not is_growing:
            return OOMPrediction(
                seconds_low=None,
                seconds_high=None,
                confidence=0.8,
                severity=Severity.WARNING,
                display=f"Low VRAM ({utilization:.0f}%)",
                utilization_pct=utilization,
            )

        # Not growing — stable
        if not is_growing:
            return OOMPrediction(
                None, None, 0.0, Severity.NONE, "Stable",
                utilization_pct=utilization,
            )

        # Rule 1: Need sustained growth
        if self._growing_samples < self.min_sustained_samples:
            return OOMPrediction(
                None, None, 0.0, Severity.NONE, "Stable",
                utilization_pct=utilization,
            )

        # Rule 2: Don't predict OOM if plenty of room
        if utilization < _MIN_UTILIZATION_PCT:
            return OOMPrediction(
                None, None, 0.0, Severity.NONE, "Stable",
                utilization_pct=utilization,
            )

        # Compute confidence: ramps with sustained samples
        confidence = min(1.0, self._growing_samples / 30.0)

        # Rule 3: Require minimum confidence
        if confidence < _MIN_CONFIDENCE:
            return OOMPrediction(
                None, None, confidence, Severity.NONE, "Stable",
                utilization_pct=utilization,
            )

        # Compute time-to-OOM range using tracked min/max rates
        min_rate = self._growing_min_rate
        max_rate = self._growing_max_rate
        if min_rate is None or max_rate is None:
            return OOMPrediction(
                None, None, 0.0, Severity.NONE, "Stable",
                utilization_pct=utilization,
            )

        min_rate_c = max(min_rate, 0.001)
        max_rate_c = max(max_rate, 0.001)

        seconds_low = free_mb / max_rate_c
        seconds_high = free_mb / min_rate_c

        if seconds_low > seconds_high:
            seconds_low, seconds_high = seconds_high, seconds_low

        rate_range = (min_rate, max_rate)

        # Rule 6: Cap at 1 hour
        if seconds_low > _CAP_SECONDS:
            return OOMPrediction(
                None, None, confidence, Severity.NONE, "OOM in >1h",
                rate_range_mb_per_sec=rate_range,
                utilization_pct=utilization,
            )

        seconds_high = min(seconds_high, _CAP_SECONDS)

        # Determine severity
        severity = Severity.WARNING
        if seconds_low < _CRITICAL_SECONDS:
            severity = Severity.CRITICAL

        display = _format_display(seconds_low, seconds_high)

        return OOMPrediction(
            seconds_low=seconds_low,
            seconds_high=seconds_high,
            confidence=confidence,
            severity=severity,
            display=display,
            rate_range_mb_per_sec=rate_range,
            utilization_pct=utilization,
        )


def _format_display(low: float, high: float) -> str:
    """Format the OOM time range for display."""
    low_s = _format_seconds(low)
    high_s = _format_seconds(high)
    if low_s == high_s:
        return f"OOM in ~{low_s}"
    return f"OOM in ~{low_s}-{high_s}"


def _format_seconds(seconds: float) -> str:
    """Format seconds into a human-readable string."""
    if seconds < 60:
        return f"{int(seconds)}s"
    minutes = int(seconds / 60)
    if minutes < 60:
        return f"{minutes}m"
    return f"{int(seconds / 3600)}h"
