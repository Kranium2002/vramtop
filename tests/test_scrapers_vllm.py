"""Tests for vLLM scraper — Prometheus text parsing."""

from __future__ import annotations

import pytest

from vramtop.enrichment.scraper import ScrapeFailedError
from vramtop.enrichment.scrapers.vllm import VLLMScraper

SAMPLE_METRICS = b"""\
# HELP vllm:gpu_cache_usage_perc GPU cache usage percentage
# TYPE vllm:gpu_cache_usage_perc gauge
vllm:gpu_cache_usage_perc 0.42
# HELP vllm:num_requests_running Number of running requests
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running 3
# HELP vllm:num_requests_waiting Number of waiting requests
# TYPE vllm:num_requests_waiting gauge
vllm:num_requests_waiting 7
# HELP vllm:avg_generation_throughput_toks_per_s Average generation throughput
# TYPE vllm:avg_generation_throughput_toks_per_s gauge
vllm:avg_generation_throughput_toks_per_s 128.5
# HELP vllm:avg_prompt_throughput_toks_per_s Average prompt throughput
# TYPE vllm:avg_prompt_throughput_toks_per_s gauge
vllm:avg_prompt_throughput_toks_per_s 256.0
# HELP some_other_metric Not relevant
# TYPE some_other_metric gauge
some_other_metric 999
"""


class TestVLLMScraper:
    def test_parse_full_metrics(self) -> None:
        scraper = VLLMScraper()
        result = scraper._parse_response(SAMPLE_METRICS)

        assert result["gpu_cache_usage_perc"] == pytest.approx(0.42)
        assert result["num_requests_running"] == pytest.approx(3.0)
        assert result["num_requests_waiting"] == pytest.approx(7.0)
        assert result["avg_generation_throughput_toks_per_s"] == pytest.approx(128.5)
        assert result["avg_prompt_throughput_toks_per_s"] == pytest.approx(256.0)
        # Unrelated metric should NOT be present
        assert "some_other_metric" not in result

    def test_parse_empty_metrics(self) -> None:
        scraper = VLLMScraper()
        result = scraper._parse_response(b"")
        assert result == {}

    def test_parse_comments_only(self) -> None:
        scraper = VLLMScraper()
        result = scraper._parse_response(b"# TYPE foo gauge\n# HELP foo desc\n")
        assert result == {}

    def test_parse_single_metric(self) -> None:
        scraper = VLLMScraper()
        result = scraper._parse_response(b"vllm:gpu_cache_usage_perc 0.85\n")
        assert result == {"gpu_cache_usage_perc": pytest.approx(0.85)}

    def test_parse_metric_with_labels(self) -> None:
        scraper = VLLMScraper()
        raw = b'vllm:num_requests_running{model="llama"} 5\n'
        result = scraper._parse_response(raw)
        assert result["num_requests_running"] == pytest.approx(5.0)

    def test_parse_invalid_utf8(self) -> None:
        scraper = VLLMScraper()
        with pytest.raises(ScrapeFailedError, match="Invalid UTF-8"):
            scraper._parse_response(b"\xff\xfe invalid")

    def test_parse_nan_value(self) -> None:
        scraper = VLLMScraper()
        result = scraper._parse_response(b"vllm:gpu_cache_usage_perc NaN\n")
        # NaN is a valid float
        import math
        assert math.isnan(result["gpu_cache_usage_perc"])

    def test_parse_scientific_notation(self) -> None:
        scraper = VLLMScraper()
        result = scraper._parse_response(b"vllm:avg_generation_throughput_toks_per_s 1.5e2\n")
        assert result["avg_generation_throughput_toks_per_s"] == pytest.approx(150.0)

    def test_endpoint_is_metrics(self) -> None:
        scraper = VLLMScraper()
        assert scraper.endpoint == "/metrics"
