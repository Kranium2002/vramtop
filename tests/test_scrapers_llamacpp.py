"""Tests for llama.cpp scraper — Prometheus /metrics parsing."""

from __future__ import annotations

import pytest

from vramtop.enrichment.scraper import ScrapeFailedError
from vramtop.enrichment.scrapers.llamacpp import LlamaCppScraper

SAMPLE_METRICS = b"""\
# HELP llamacpp_kv_cache_usage KV cache usage ratio
# TYPE llamacpp_kv_cache_usage gauge
llamacpp_kv_cache_usage 0.75
# HELP llamacpp_requests_processing Number of requests being processed
# TYPE llamacpp_requests_processing gauge
llamacpp_requests_processing 2
# HELP llamacpp_requests_pending Number of pending requests
# TYPE llamacpp_requests_pending gauge
llamacpp_requests_pending 5
# HELP llamacpp_tokens_predicted_total Total predicted tokens
# TYPE llamacpp_tokens_predicted_total counter
llamacpp_tokens_predicted_total 12345
# HELP unrelated_metric Not relevant
# TYPE unrelated_metric gauge
unrelated_metric 999
"""


class TestLlamaCppScraper:
    def test_parse_full_metrics(self) -> None:
        scraper = LlamaCppScraper()
        result = scraper._parse_response(SAMPLE_METRICS)

        assert result["kv_cache_usage"] == pytest.approx(0.75)
        assert result["requests_processing"] == pytest.approx(2.0)
        assert result["requests_pending"] == pytest.approx(5.0)
        assert result["tokens_predicted_total"] == pytest.approx(12345.0)
        assert "unrelated_metric" not in result

    def test_parse_empty_metrics(self) -> None:
        scraper = LlamaCppScraper()
        result = scraper._parse_response(b"")
        assert result == {}

    def test_parse_single_metric(self) -> None:
        scraper = LlamaCppScraper()
        result = scraper._parse_response(b"llamacpp_kv_cache_usage 0.33\n")
        assert result == {"kv_cache_usage": pytest.approx(0.33)}

    def test_parse_metric_with_labels(self) -> None:
        scraper = LlamaCppScraper()
        raw = b'llamacpp_requests_processing{slot="0"} 1\n'
        result = scraper._parse_response(raw)
        assert result["requests_processing"] == pytest.approx(1.0)

    def test_parse_invalid_utf8(self) -> None:
        scraper = LlamaCppScraper()
        with pytest.raises(ScrapeFailedError, match="Invalid UTF-8"):
            scraper._parse_response(b"\xff\xfe bad")

    def test_parse_scientific_notation(self) -> None:
        scraper = LlamaCppScraper()
        result = scraper._parse_response(b"llamacpp_tokens_predicted_total 1.5e4\n")
        assert result["tokens_predicted_total"] == pytest.approx(15000.0)

    def test_endpoint_is_metrics(self) -> None:
        scraper = LlamaCppScraper()
        assert scraper.endpoint == "/metrics"

    def test_comments_only(self) -> None:
        scraper = LlamaCppScraper()
        result = scraper._parse_response(b"# TYPE foo gauge\n# HELP foo desc\n")
        assert result == {}
