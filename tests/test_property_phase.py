"""Hypothesis property-based tests for the phase detector."""

from __future__ import annotations

from hypothesis import given, settings, strategies as st

from vramtop.analysis.phase_detector import Phase, PhaseDetector


@given(
    deltas=st.lists(
        st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
        min_size=0,
        max_size=200,
    )
)
@settings(max_examples=200)
def test_phase_detector_never_crashes(deltas: list[float]) -> None:
    """PhaseDetector.update should never crash on arbitrary float deltas."""
    detector = PhaseDetector()
    for delta in deltas:
        result = detector.update(delta, 1.0)
        assert isinstance(result.phase, Phase)


@given(
    deltas=st.lists(
        st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
        min_size=0,
        max_size=200,
    )
)
@settings(max_examples=200)
def test_confidence_always_bounded(deltas: list[float]) -> None:
    """Confidence should always be in [0, 1]."""
    detector = PhaseDetector()
    for delta in deltas:
        result = detector.update(delta, 1.0)
        assert 0.0 <= result.confidence <= 1.0


@given(
    deltas=st.lists(
        st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
        min_size=0,
        max_size=200,
    ),
    dt=st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=200)
def test_phase_detector_handles_varying_dt(deltas: list[float], dt: float) -> None:
    """PhaseDetector should handle any non-negative dt without crashing."""
    detector = PhaseDetector()
    for delta in deltas:
        result = detector.update(delta, dt)
        assert isinstance(result.phase, Phase)
        assert 0.0 <= result.confidence <= 1.0
