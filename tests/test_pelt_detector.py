"""Tests for PELT changepoint detector."""

from __future__ import annotations

import importlib
import sys
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from vramtop.analysis.phase_detector import Phase
from vramtop.analysis.pelt_detector import (
    PENALTY_PRESETS,
    PELTDetector,
    get_preset_penalty,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _step_timeseries(
    levels: list[float], segment_length: int = 50
) -> list[float]:
    """Build a timeseries that holds each level for *segment_length* samples."""
    ts: list[float] = []
    for level in levels:
        ts.extend([level] * segment_length)
    return ts


# ---------------------------------------------------------------------------
# Penalty presets
# ---------------------------------------------------------------------------


class TestPenaltyPresets:
    def test_known_presets(self) -> None:
        assert get_preset_penalty("pytorch_training") == 10.0
        assert get_preset_penalty("inference_server") == 50.0
        assert get_preset_penalty("unknown") == 20.0

    def test_framework_name_mapping(self) -> None:
        """Enrichment framework names map to correct presets."""
        assert get_preset_penalty("pytorch") == 10.0  # -> pytorch_training
        assert get_preset_penalty("vllm") == 50.0  # -> inference_server
        assert get_preset_penalty("sglang") == 50.0  # -> inference_server
        assert get_preset_penalty("ollama") == 50.0  # -> inference_server
        assert get_preset_penalty("llamacpp") == 50.0  # -> inference_server
        assert get_preset_penalty("tgi") == 50.0  # -> inference_server
        assert get_preset_penalty("jax") == 10.0  # -> pytorch_training

    def test_none_returns_unknown(self) -> None:
        assert get_preset_penalty(None) == 20.0

    def test_unrecognised_framework_returns_unknown(self) -> None:
        assert get_preset_penalty("my_custom_framework") == 20.0

    def test_preset_dict_contents(self) -> None:
        assert set(PENALTY_PRESETS.keys()) == {
            "pytorch_training",
            "inference_server",
            "unknown",
        }


# ---------------------------------------------------------------------------
# Empty / short timeseries
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_timeseries_returns_empty(self) -> None:
        detector = PELTDetector()
        assert detector.detect_changepoints([]) == []

    def test_short_timeseries_returns_empty(self) -> None:
        detector = PELTDetector(min_size=10)
        assert detector.detect_changepoints([1.0] * 5) == []

    def test_exactly_min_size_does_not_crash(self) -> None:
        detector = PELTDetector(min_size=10)
        # Should not raise — may or may not find changepoints
        result = detector.detect_changepoints([1.0] * 10)
        assert isinstance(result, list)

    def test_classify_empty_timeseries(self) -> None:
        detector = PELTDetector()
        assert detector.classify_segments([], []) == []

    def test_classify_single_point(self) -> None:
        detector = PELTDetector()
        assert detector.classify_segments([5.0], []) == []


# ---------------------------------------------------------------------------
# Segment classification
# ---------------------------------------------------------------------------


class TestSegmentClassification:
    def test_stable_segment(self) -> None:
        detector = PELTDetector()
        ts = [100.0] * 20
        phases = detector.classify_segments(ts, [])
        assert phases == [Phase.STABLE]

    def test_growing_segment(self) -> None:
        detector = PELTDetector()
        ts = [float(i * 10) for i in range(20)]
        phases = detector.classify_segments(ts, [])
        assert phases == [Phase.GROWING]

    def test_shrinking_segment(self) -> None:
        detector = PELTDetector()
        ts = [float(200 - i * 10) for i in range(20)]
        phases = detector.classify_segments(ts, [])
        assert phases == [Phase.SHRINKING]

    def test_multiple_segments(self) -> None:
        detector = PELTDetector()
        # Segment 0-9: growing,  10-19: stable
        ts = [float(i * 10) for i in range(10)] + [90.0] * 10
        phases = detector.classify_segments(ts, [10])
        assert len(phases) == 2
        assert phases[0] == Phase.GROWING
        assert phases[1] == Phase.STABLE

    def test_three_segments_grow_stable_shrink(self) -> None:
        detector = PELTDetector()
        growing = [float(i * 10) for i in range(10)]
        stable = [90.0] * 10
        shrinking = [float(90 - i * 10) for i in range(10)]
        ts = growing + stable + shrinking
        phases = detector.classify_segments(ts, [10, 20])
        assert phases == [Phase.GROWING, Phase.STABLE, Phase.SHRINKING]


# ---------------------------------------------------------------------------
# Changepoint detection with ruptures (mock)
# ---------------------------------------------------------------------------


class TestDetectWithMockRuptures:
    """Test changepoint detection by mocking the ruptures library."""

    def test_step_function_finds_changepoint(self) -> None:
        """A clear step function should produce at least one changepoint."""
        detector = PELTDetector(penalty=5.0, min_size=5)
        ts = _step_timeseries([100.0, 500.0], segment_length=30)

        # Mock ruptures if not installed
        mock_algo = MagicMock()
        mock_algo.predict.return_value = [30, len(ts)]

        mock_pelt_cls = MagicMock()
        mock_pelt_cls.return_value.fit.return_value = mock_algo

        with patch("vramtop.analysis.pelt_detector._HAS_RUPTURES", True), \
             patch("vramtop.analysis.pelt_detector.ruptures", create=True) as mock_rup:
            mock_rup.Pelt.return_value.fit.return_value = mock_algo
            result = detector.detect_changepoints(ts)

        assert 30 in result
        # Terminal index should be stripped
        assert len(ts) not in result

    def test_constant_timeseries_no_changepoints(self) -> None:
        """A flat timeseries should produce no changepoints."""
        detector = PELTDetector(penalty=20.0, min_size=5)
        ts = [100.0] * 50

        mock_algo = MagicMock()
        mock_algo.predict.return_value = [len(ts)]

        with patch("vramtop.analysis.pelt_detector._HAS_RUPTURES", True), \
             patch("vramtop.analysis.pelt_detector.ruptures", create=True) as mock_rup:
            mock_rup.Pelt.return_value.fit.return_value = mock_algo
            result = detector.detect_changepoints(ts)

        assert result == []

    def test_multiple_changepoints(self) -> None:
        """A multi-step timeseries should return multiple changepoints."""
        detector = PELTDetector(penalty=5.0, min_size=5)
        ts = _step_timeseries([100.0, 300.0, 500.0], segment_length=20)

        mock_algo = MagicMock()
        mock_algo.predict.return_value = [20, 40, len(ts)]

        with patch("vramtop.analysis.pelt_detector._HAS_RUPTURES", True), \
             patch("vramtop.analysis.pelt_detector.ruptures", create=True) as mock_rup:
            mock_rup.Pelt.return_value.fit.return_value = mock_algo
            result = detector.detect_changepoints(ts)

        assert result == [20, 40]


# ---------------------------------------------------------------------------
# Import fallback
# ---------------------------------------------------------------------------


class TestImportFallback:
    """Verify graceful degradation when ruptures is not installed."""

    def test_detect_returns_empty_when_ruptures_missing(self) -> None:
        detector = PELTDetector()
        with patch("vramtop.analysis.pelt_detector._HAS_RUPTURES", False):
            result = detector.detect_changepoints([1.0] * 50)
        assert result == []

    def test_classify_still_works_without_ruptures(self) -> None:
        """classify_segments does not depend on ruptures at all."""
        detector = PELTDetector()
        ts = [float(i) for i in range(20)]
        with patch("vramtop.analysis.pelt_detector._HAS_RUPTURES", False):
            phases = detector.classify_segments(ts, [10])
        assert len(phases) == 2

    def test_module_import_with_missing_ruptures(self) -> None:
        """Importing the module should not raise even if ruptures is absent."""
        # Temporarily remove ruptures from sys.modules to simulate absence
        saved: dict[str, Any] = {}
        modules_to_remove = [k for k in sys.modules if k.startswith("ruptures")]
        for mod_name in modules_to_remove:
            saved[mod_name] = sys.modules.pop(mod_name)

        # Also hide pelt_detector so it re-imports
        pelt_mod_key = "vramtop.analysis.pelt_detector"
        saved_pelt = sys.modules.pop(pelt_mod_key, None)

        try:
            with patch.dict(sys.modules, {"ruptures": None}):
                mod = importlib.import_module(pelt_mod_key)
                assert not mod._HAS_RUPTURES
        finally:
            # Restore
            for mod_name, mod_obj in saved.items():
                sys.modules[mod_name] = mod_obj
            if saved_pelt is not None:
                sys.modules[pelt_mod_key] = saved_pelt


# ---------------------------------------------------------------------------
# min_size enforcement
# ---------------------------------------------------------------------------


class TestMinSizeEnforcement:
    def test_min_size_5_allows_5_samples(self) -> None:
        detector = PELTDetector(min_size=5)
        ts = [1.0] * 5

        mock_algo = MagicMock()
        mock_algo.predict.return_value = [len(ts)]

        with patch("vramtop.analysis.pelt_detector._HAS_RUPTURES", True), \
             patch("vramtop.analysis.pelt_detector.ruptures", create=True) as mock_rup:
            mock_rup.Pelt.return_value.fit.return_value = mock_algo
            result = detector.detect_changepoints(ts)

        assert isinstance(result, list)

    def test_min_size_5_rejects_4_samples(self) -> None:
        detector = PELTDetector(min_size=5)
        result = detector.detect_changepoints([1.0] * 4)
        assert result == []

    def test_min_size_passed_to_ruptures(self) -> None:
        """Verify min_size is forwarded to ruptures.Pelt constructor."""
        detector = PELTDetector(min_size=7)
        ts = [1.0] * 20

        mock_algo = MagicMock()
        mock_algo.predict.return_value = [len(ts)]

        with patch("vramtop.analysis.pelt_detector._HAS_RUPTURES", True), \
             patch("vramtop.analysis.pelt_detector.ruptures", create=True) as mock_rup:
            mock_rup.Pelt.return_value.fit.return_value = mock_algo
            detector.detect_changepoints(ts)

        mock_rup.Pelt.assert_called_once_with(model="l2", min_size=7)
