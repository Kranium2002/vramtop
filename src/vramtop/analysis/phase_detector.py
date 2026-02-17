"""Variance-threshold phase detector for GPU memory timeseries."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum


class Phase(Enum):
    STABLE = "stable"
    GROWING = "growing"
    SHRINKING = "shrinking"
    VOLATILE = "volatile"


@dataclass
class PhaseState:
    phase: Phase
    duration_samples: int
    rate_mb_per_sec: float  # EMA-smoothed within this phase
    confidence: float  # 0.0 - 1.0


class PhaseDetector:
    """Sliding-window phase detection for GPU memory timeseries.

    Algorithm:
    1. Maintain rolling window of memory deltas (size=W, default 10)
    2. Compute rolling variance and mean of deltas
    3. Classify:
       - variance < threshold AND |mean| < noise_floor -> STABLE
       - mean > noise_floor AND >60% deltas positive -> GROWING
       - mean < -noise_floor AND >60% deltas negative -> SHRINKING
       - else -> VOLATILE
    4. Hysteresis: require M consecutive samples (default 3)
       before transitioning, preventing flicker.
    """

    def __init__(
        self,
        window_size: int = 10,
        hysteresis_samples: int = 3,
        noise_floor_mb: float = 1.0,
        variance_threshold: float = 4.0,
        ema_alpha: float = 0.3,
    ) -> None:
        self.window: deque[float] = deque(maxlen=window_size)
        self.hysteresis = hysteresis_samples
        self.noise_floor = noise_floor_mb
        self.var_threshold = variance_threshold
        self.ema_alpha = ema_alpha

        self._current_phase = Phase.STABLE
        self._candidate_phase = Phase.STABLE
        self._candidate_count = 0
        self._phase_duration = 0
        self._ema_rate = 0.0

    def update(self, delta_mb: float, dt_seconds: float) -> PhaseState:
        rate = delta_mb / max(dt_seconds, 0.001)
        self.window.append(rate)

        if len(self.window) < 3:
            return PhaseState(Phase.STABLE, 0, 0.0, 0.0)

        rates = list(self.window)
        mean_rate = sum(rates) / len(rates)
        variance = sum((r - mean_rate) ** 2 for r in rates) / len(rates)
        positive_frac = sum(1 for r in rates if r > self.noise_floor) / len(rates)
        negative_frac = sum(1 for r in rates if r < -self.noise_floor) / len(rates)

        if variance < self.var_threshold and abs(mean_rate) < self.noise_floor:
            detected = Phase.STABLE
        elif mean_rate > self.noise_floor and positive_frac > 0.6:
            detected = Phase.GROWING
        elif mean_rate < -self.noise_floor and negative_frac > 0.6:
            detected = Phase.SHRINKING
        else:
            detected = Phase.VOLATILE

        if detected == self._candidate_phase:
            self._candidate_count += 1
        else:
            self._candidate_phase = detected
            self._candidate_count = 1

        if (
            self._candidate_count >= self.hysteresis
            and self._current_phase != self._candidate_phase
        ):
            self._current_phase = self._candidate_phase
            self._phase_duration = 0
            self._ema_rate = mean_rate

        self._phase_duration += 1
        self._ema_rate = self.ema_alpha * rate + (1 - self.ema_alpha) * self._ema_rate

        confidence = min(1.0, self._phase_duration / 30.0) * (
            1.0 / (1.0 + variance / 100.0)
        )

        return PhaseState(
            phase=self._current_phase,
            duration_samples=self._phase_duration,
            rate_mb_per_sec=self._ema_rate,
            confidence=confidence,
        )
