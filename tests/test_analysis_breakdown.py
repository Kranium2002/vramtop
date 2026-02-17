"""Tests for memory breakdown estimation."""

from __future__ import annotations

import pytest

from vramtop.analysis.breakdown import MemoryBreakdown, estimate_breakdown


class TestEstimateBreakdown:
    """Test weight vs dynamic memory estimation."""

    def test_no_model_files(self):
        result = estimate_breakdown(total_used=1_000_000, model_file_sizes=[])
        assert result.estimated_weights_bytes is None
        assert result.estimated_dynamic_bytes is None
        assert result.confidence == 0.0
        assert "estimate" in result.label

    def test_zero_total(self):
        result = estimate_breakdown(total_used=0, model_file_sizes=[500])
        assert result.estimated_weights_bytes is None
        assert result.confidence == 0.0
        assert "estimate" in result.label

    def test_negative_total(self):
        result = estimate_breakdown(total_used=-100, model_file_sizes=[500])
        assert result.total_bytes == 0
        assert "estimate" in result.label

    def test_basic_breakdown(self):
        # 1GB total, 700MB in weights
        total = 1_000_000_000
        weights = [700_000_000]
        result = estimate_breakdown(total, weights)

        assert result.total_bytes == total
        assert result.estimated_weights_bytes == 700_000_000
        assert result.estimated_dynamic_bytes == 300_000_000
        assert result.confidence > 0.0
        assert "estimate" in result.label

    def test_weights_clamped_to_total(self):
        # Model files larger than total GPU usage (e.g., quantized in VRAM)
        total = 500_000_000
        weights = [800_000_000]
        result = estimate_breakdown(total, weights)

        assert result.estimated_weights_bytes == total
        assert result.estimated_dynamic_bytes == 0

    def test_multiple_model_files(self):
        total = 10_000_000_000
        weights = [3_000_000_000, 3_000_000_000, 1_000_000_000]
        result = estimate_breakdown(total, weights)

        assert result.estimated_weights_bytes == 7_000_000_000
        assert result.estimated_dynamic_bytes == 3_000_000_000

    def test_label_always_contains_estimate(self):
        # With data
        r1 = estimate_breakdown(1000, [500])
        assert "estimate" in r1.label

        # Without data
        r2 = estimate_breakdown(1000, [])
        assert "estimate" in r2.label

        # Zero total
        r3 = estimate_breakdown(0, [])
        assert "estimate" in r3.label

    def test_confidence_moderate_for_typical_ratio(self):
        # 60% weights -> moderate confidence (file-size estimates are imprecise)
        total = 10_000
        weights = [6_000]
        result = estimate_breakdown(total, weights)
        assert result.confidence >= 0.4
        # Capped at 0.5 — file sizes don't reliably reflect in-memory sizes
        assert result.confidence <= 0.6

    def test_confidence_lower_for_extreme_ratio(self):
        # 5% weights -> low confidence
        total = 10_000
        weights = [500]
        result = estimate_breakdown(total, weights)
        assert result.confidence < 0.4
