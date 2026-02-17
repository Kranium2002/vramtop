"""Hypothesis property-based tests for the GPU-level OOM predictor."""

from __future__ import annotations

from hypothesis import given, settings, strategies as st

from vramtop.analysis.oom_predictor import OOMPredictor, Severity


@given(
    deltas=st.lists(
        st.floats(min_value=-500.0, max_value=500.0, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=100,
    ),
    total_mb=st.floats(min_value=1000.0, max_value=100000.0, allow_nan=False, allow_infinity=False),
    used_frac=st.floats(min_value=0.01, max_value=0.99, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=200)
def test_oom_range_always_valid(
    deltas: list[float],
    total_mb: float,
    used_frac: float,
) -> None:
    """When low and high are both present, low <= high must hold."""
    predictor = OOMPredictor(min_sustained_samples=3, min_rate_mb_per_sec=1.0)
    used_mb = total_mb * used_frac
    free_mb = total_mb - used_mb

    for delta in deltas:
        result = predictor.update(
            used_mb=used_mb,
            free_mb=max(free_mb, 0.0),
            total_mb=total_mb,
            delta_mb=delta,
            dt_seconds=1.0,
        )

        if result.seconds_low is not None and result.seconds_high is not None:
            assert result.seconds_low <= result.seconds_high, (
                f"low={result.seconds_low} > high={result.seconds_high}"
            )


@given(
    deltas=st.lists(
        st.floats(min_value=-500.0, max_value=500.0, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=100,
    ),
    total_mb=st.floats(min_value=1000.0, max_value=100000.0, allow_nan=False, allow_infinity=False),
    used_frac=st.floats(min_value=0.01, max_value=0.99, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=200)
def test_oom_confidence_always_bounded(
    deltas: list[float],
    total_mb: float,
    used_frac: float,
) -> None:
    """Confidence should always be in [0, 1]."""
    predictor = OOMPredictor(min_sustained_samples=3, min_rate_mb_per_sec=1.0)
    used_mb = total_mb * used_frac
    free_mb = total_mb - used_mb

    for delta in deltas:
        result = predictor.update(
            used_mb=used_mb,
            free_mb=max(free_mb, 0.0),
            total_mb=total_mb,
            delta_mb=delta,
            dt_seconds=1.0,
        )
        assert 0.0 <= result.confidence <= 1.0


@given(
    total_mb=st.floats(min_value=4000.0, max_value=100000.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=50)
def test_idle_gpu_never_alerts(total_mb: float) -> None:
    """An idle GPU should never produce OOM alerts regardless of total memory."""
    predictor = OOMPredictor()
    for _ in range(30):
        result = predictor.update(
            used_mb=10.0,
            free_mb=total_mb - 10.0,
            total_mb=total_mb,
            delta_mb=0.0,
            dt_seconds=1.0,
        )
    assert result.severity == Severity.NONE
