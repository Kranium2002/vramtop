"""Tests for the GPU-level OOM predictor."""

from __future__ import annotations

from vramtop.analysis.oom_predictor import OOMPredictor, Severity


def _mb(gb: float) -> float:
    return gb * 1024.0


class TestOOMPredictor:
    """Test suite for OOMPredictor."""

    def test_stable_when_no_growth(self) -> None:
        """Should show Stable when GPU memory is not growing."""
        predictor = OOMPredictor()
        result = predictor.update(
            used_mb=1000, free_mb=15000, total_mb=16000,
            delta_mb=0.0, dt_seconds=1.0,
        )
        assert result.severity == Severity.NONE
        assert result.display == "Stable"
        assert result.seconds_low is None

    def test_no_prediction_under_min_samples(self) -> None:
        """Should not predict until growth sustained for min_sustained_samples."""
        predictor = OOMPredictor(min_sustained_samples=10, min_rate_mb_per_sec=5.0)
        # Feed 5 growing samples — not enough
        for _ in range(5):
            result = predictor.update(
                used_mb=10000, free_mb=6000, total_mb=16000,
                delta_mb=100.0, dt_seconds=1.0,
            )
        assert result.severity == Severity.NONE

    def test_no_prediction_low_utilization(self) -> None:
        """Should not predict OOM when <50% utilization even if growing."""
        predictor = OOMPredictor(min_sustained_samples=3, min_rate_mb_per_sec=5.0)
        # Growing fast but only 20% utilized — no concern
        for _ in range(20):
            result = predictor.update(
                used_mb=3200, free_mb=12800, total_mb=16000,
                delta_mb=50.0, dt_seconds=1.0,
            )
        assert result.severity == Severity.NONE
        assert result.display == "Stable"

    def test_warning_when_growing_and_high_util(self) -> None:
        """Should warn when sustained growth with >50% utilization."""
        predictor = OOMPredictor(min_sustained_samples=5, min_rate_mb_per_sec=5.0)
        # 60% utilized and growing at 50 MB/s
        for _ in range(20):
            result = predictor.update(
                used_mb=9600, free_mb=6400, total_mb=16000,
                delta_mb=50.0, dt_seconds=1.0,
            )
        assert result.severity in (Severity.WARNING, Severity.CRITICAL)
        assert result.seconds_low is not None
        assert "OOM" in result.display

    def test_critical_when_near_oom(self) -> None:
        """Should show CRITICAL severity when <5 min to OOM."""
        predictor = OOMPredictor(min_sustained_samples=5, min_rate_mb_per_sec=5.0)
        # 90% utilized, only 1600 MB free, growing at 100 MB/s → ~16s
        for _ in range(20):
            result = predictor.update(
                used_mb=14400, free_mb=1600, total_mb=16000,
                delta_mb=100.0, dt_seconds=1.0,
            )
        assert result.severity == Severity.CRITICAL
        assert result.seconds_low is not None
        assert result.seconds_low < 300

    def test_range_not_point_estimate(self) -> None:
        """Predictions should produce a range (low <= high)."""
        predictor = OOMPredictor(min_sustained_samples=3, min_rate_mb_per_sec=5.0)
        # Feed varying growth rates to create a range
        rates = [50.0, 30.0, 80.0, 40.0, 60.0, 50.0, 70.0, 45.0, 55.0, 65.0,
                 50.0, 50.0, 50.0, 50.0, 50.0]
        for rate in rates:
            result = predictor.update(
                used_mb=12000, free_mb=4000, total_mb=16000,
                delta_mb=rate, dt_seconds=1.0,
            )
        if result.seconds_low is not None and result.seconds_high is not None:
            assert result.seconds_low <= result.seconds_high

    def test_suppressed_below_rate_threshold(self) -> None:
        """Should suppress prediction if EMA rate < min_rate_mb_per_sec."""
        predictor = OOMPredictor(min_sustained_samples=3, min_rate_mb_per_sec=10.0)
        # Growing at only 2 MB/s — below threshold
        for _ in range(20):
            result = predictor.update(
                used_mb=12000, free_mb=4000, total_mb=16000,
                delta_mb=2.0, dt_seconds=1.0,
            )
        assert result.severity == Severity.NONE

    def test_capped_at_one_hour(self) -> None:
        """Should cap prediction at '>1h' for very slow growth with high utilization."""
        # 55% utilized, large free pool relative to rate → >1h prediction
        # used=55000, free=45000, total=100000 → 55% util, 45000/6 = 7500s > 3600
        predictor = OOMPredictor(min_sustained_samples=3, min_rate_mb_per_sec=5.0)
        for _ in range(20):
            result = predictor.update(
                used_mb=55000, free_mb=45000, total_mb=100000,
                delta_mb=6.0, dt_seconds=1.0,
            )
        assert "1h" in result.display

    def test_resets_when_growth_stops(self) -> None:
        """Rate tracking should reset when growth stops and resumes."""
        predictor = OOMPredictor(min_sustained_samples=3, min_rate_mb_per_sec=5.0)
        # Build up growing state
        for _ in range(15):
            predictor.update(
                used_mb=12000, free_mb=4000, total_mb=16000,
                delta_mb=100.0, dt_seconds=1.0,
            )
        # Stop growing
        for _ in range(10):
            predictor.update(
                used_mb=12000, free_mb=4000, total_mb=16000,
                delta_mb=0.0, dt_seconds=1.0,
            )
        # Resume — should need sustained samples again
        result = predictor.update(
            used_mb=12000, free_mb=4000, total_mb=16000,
            delta_mb=50.0, dt_seconds=1.0,
        )
        # Just one sample after reset, EMA still low from zeros
        assert result.severity == Severity.NONE

    def test_high_utilization_warning_without_growth(self) -> None:
        """Should show 'Low VRAM' warning at >85% even without growth."""
        predictor = OOMPredictor()
        # 90% utilized but not growing
        for _ in range(5):
            result = predictor.update(
                used_mb=14400, free_mb=1600, total_mb=16000,
                delta_mb=0.0, dt_seconds=1.0,
            )
        assert result.severity == Severity.WARNING
        assert "Low VRAM" in result.display

    def test_no_false_alarm_on_idle_gpu(self) -> None:
        """An idle GPU (near-zero usage) should never trigger OOM alerts."""
        predictor = OOMPredictor()
        for _ in range(30):
            result = predictor.update(
                used_mb=2, free_mb=15998, total_mb=16000,
                delta_mb=0.1, dt_seconds=1.0,
            )
        assert result.severity == Severity.NONE
        assert result.display == "Stable"

    def test_utilization_pct_reported(self) -> None:
        """Predictions should report utilization percentage."""
        predictor = OOMPredictor()
        result = predictor.update(
            used_mb=8000, free_mb=8000, total_mb=16000,
            delta_mb=0.0, dt_seconds=1.0,
        )
        assert abs(result.utilization_pct - 50.0) < 0.1
