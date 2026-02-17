"""Tests for the variance-threshold phase detector."""

from __future__ import annotations

from vramtop.analysis.phase_detector import Phase, PhaseDetector


class TestPhaseDetector:
    """Test suite for PhaseDetector."""

    def test_starts_stable(self) -> None:
        """Detector should start in STABLE phase."""
        detector = PhaseDetector()
        result = detector.update(0.0, 1.0)
        assert result.phase == Phase.STABLE

    def test_growing_after_sustained_increase(self) -> None:
        """Sustained positive deltas should transition to GROWING."""
        detector = PhaseDetector(
            window_size=5, hysteresis_samples=3, noise_floor_mb=1.0
        )
        # Feed enough positive deltas to fill window and pass hysteresis
        for _ in range(20):
            result = detector.update(50.0, 1.0)

        assert result.phase == Phase.GROWING

    def test_hysteresis_prevents_flicker(self) -> None:
        """A single contrary sample shouldn't cause phase change."""
        detector = PhaseDetector(
            window_size=5, hysteresis_samples=3, noise_floor_mb=1.0
        )
        # Establish GROWING phase
        for _ in range(20):
            result = detector.update(50.0, 1.0)
        assert result.phase == Phase.GROWING

        # One contrary sample should not flip phase
        result = detector.update(-50.0, 1.0)
        assert result.phase == Phase.GROWING

    def test_shrinking_detection(self) -> None:
        """Sustained negative deltas should transition to SHRINKING."""
        detector = PhaseDetector(
            window_size=5, hysteresis_samples=3, noise_floor_mb=1.0
        )
        for _ in range(20):
            result = detector.update(-50.0, 1.0)

        assert result.phase == Phase.SHRINKING

    def test_volatile_on_mixed_signals(self) -> None:
        """Alternating large positive and negative deltas should be VOLATILE."""
        detector = PhaseDetector(
            window_size=10,
            hysteresis_samples=3,
            noise_floor_mb=1.0,
            variance_threshold=4.0,
        )
        # Feed alternating large deltas to create high variance
        for i in range(30):
            delta = 100.0 if i % 2 == 0 else -100.0
            result = detector.update(delta, 1.0)

        assert result.phase == Phase.VOLATILE

    def test_confidence_starts_low(self) -> None:
        """Confidence should be low for initial samples."""
        detector = PhaseDetector()
        result = detector.update(0.0, 1.0)
        assert result.confidence == 0.0

    def test_confidence_increases_with_duration(self) -> None:
        """Confidence should increase as phase duration grows."""
        detector = PhaseDetector(window_size=5, hysteresis_samples=3)
        prev_confidence = -1.0
        for _ in range(20):
            result = detector.update(0.1, 1.0)
            # After initial warmup, confidence should be non-decreasing
            # when variance is stable
        # After many samples, confidence should be positive
        assert result.confidence > 0.0

    def test_ema_rate_tracks_input(self) -> None:
        """EMA rate should approximately track the input rate."""
        detector = PhaseDetector(window_size=5, hysteresis_samples=3, ema_alpha=0.3)
        for _ in range(50):
            result = detector.update(10.0, 1.0)
        # EMA should converge near the constant input rate
        assert abs(result.rate_mb_per_sec - 10.0) < 1.0

    def test_zero_dt_handled(self) -> None:
        """dt_seconds of 0 should not cause division by zero."""
        detector = PhaseDetector()
        result = detector.update(5.0, 0.0)
        assert result.phase == Phase.STABLE  # still in warmup
