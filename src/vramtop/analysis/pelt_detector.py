"""PELT changepoint detection for GPU memory timeseries.

Wraps the ``ruptures`` library's PELT algorithm with l2 cost model.
Falls back gracefully when ``ruptures`` is not installed (optional
``pelt`` extra dependency).
"""

from __future__ import annotations

import logging

from vramtop.analysis.phase_detector import Phase

logger = logging.getLogger(__name__)

try:
    import ruptures  # type: ignore[import-not-found,import-untyped,unused-ignore]

    _HAS_RUPTURES = True
except ImportError:
    ruptures = None  # type: ignore[assignment,unused-ignore]
    _HAS_RUPTURES = False
    logger.warning(
        "ruptures not installed — PELT changepoint detection unavailable. "
        "Install with: pip install vramtop[pelt]"
    )


PENALTY_PRESETS: dict[str, float] = {
    "pytorch_training": 10.0,
    "inference_server": 50.0,
    "unknown": 20.0,
}

# Map enrichment framework names to penalty preset keys.
_FRAMEWORK_TO_PRESET: dict[str, str] = {
    "pytorch": "pytorch_training",
    "vllm": "inference_server",
    "sglang": "inference_server",
    "tgi": "inference_server",
    "ollama": "inference_server",
    "llamacpp": "inference_server",
    "jax": "pytorch_training",
}


def get_preset_penalty(framework: str | None) -> float:
    """Return a PELT penalty tuned for a given framework.

    Maps enrichment framework names (e.g. ``"pytorch"``, ``"vllm"``)
    to preset keys, then falls back to ``"unknown"`` (20.0).
    """
    if framework is None:
        return PENALTY_PRESETS["unknown"]
    preset_key = _FRAMEWORK_TO_PRESET.get(framework, framework)
    return PENALTY_PRESETS.get(preset_key, PENALTY_PRESETS["unknown"])


class PELTDetector:
    """PELT-based changepoint detector for GPU memory timeseries.

    Parameters
    ----------
    penalty:
        PELT penalty parameter — higher values yield fewer changepoints.
    min_size:
        Minimum number of samples between consecutive changepoints, and
        the minimum length of timeseries to attempt detection on.
    """

    def __init__(self, penalty: float = 20.0, min_size: int = 10) -> None:
        self.penalty = penalty
        self.min_size = min_size

    def detect_changepoints(self, timeseries: list[float]) -> list[int]:
        """Return changepoint indices in *timeseries*.

        If ``ruptures`` is not installed, or the timeseries is shorter
        than ``min_size``, an empty list is returned.
        """
        if not _HAS_RUPTURES or ruptures is None:
            return []
        if len(timeseries) < self.min_size:
            return []

        try:
            import numpy as np  # type: ignore[import-not-found,unused-ignore]
        except ImportError:
            return []

        signal = np.array(timeseries, dtype=np.float64)
        algo = ruptures.Pelt(model="l2", min_size=self.min_size).fit(signal)
        # ruptures returns breakpoint indices including the final index
        # (len(timeseries)), which we strip.
        breakpoints: list[int] = algo.predict(pen=self.penalty)
        # Remove the terminal index that ruptures always appends.
        if breakpoints and breakpoints[-1] == len(timeseries):
            breakpoints = breakpoints[:-1]
        return breakpoints

    def classify_segments(
        self, timeseries: list[float], changepoints: list[int]
    ) -> list[Phase]:
        """Classify each segment delimited by *changepoints*.

        Segments are defined as:
            [0, cp0), [cp0, cp1), ..., [cpN, len(timeseries))

        Classification uses the same logic as ``PhaseDetector``:
        compute per-segment mean delta between consecutive points, then:
        - high variance with mixed sign deltas -> VOLATILE
        - ``|mean_delta| < noise_floor`` -> STABLE
        - ``mean_delta > noise_floor`` -> GROWING
        - ``mean_delta < -noise_floor`` -> SHRINKING

        The noise floor is fixed at 1.0 MB, consistent with
        ``PhaseDetector.noise_floor_mb``.
        """
        noise_floor = 1.0
        if len(timeseries) < 2:
            return []

        boundaries = [0, *changepoints, len(timeseries)]
        phases: list[Phase] = []

        for i in range(len(boundaries) - 1):
            start = boundaries[i]
            end = boundaries[i + 1]
            segment = timeseries[start:end]

            if len(segment) < 2:
                phases.append(Phase.STABLE)
                continue

            deltas = [segment[j + 1] - segment[j] for j in range(len(segment) - 1)]
            mean_delta = sum(deltas) / len(deltas)

            # Check for VOLATILE: high variance with mixed-sign deltas.
            # A segment is volatile if both positive and negative deltas
            # are present and the variance is large relative to noise floor.
            if len(deltas) >= 3:
                pos = sum(1 for d in deltas if d > noise_floor)
                neg = sum(1 for d in deltas if d < -noise_floor)
                frac_pos = pos / len(deltas)
                frac_neg = neg / len(deltas)
                # Mixed direction: neither >60% positive nor >60% negative
                if frac_pos >= 0.2 and frac_neg >= 0.2:
                    variance = sum((d - mean_delta) ** 2 for d in deltas) / len(deltas)
                    if variance > noise_floor**2 * 4:
                        phases.append(Phase.VOLATILE)
                        continue

            if abs(mean_delta) < noise_floor:
                phases.append(Phase.STABLE)
            elif mean_delta > noise_floor:
                phases.append(Phase.GROWING)
            else:
                phases.append(Phase.SHRINKING)

        return phases
