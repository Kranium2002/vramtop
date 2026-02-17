"""Tests for SGLang scraper — JSON /get_model_info parsing."""

from __future__ import annotations

import json

import pytest

from vramtop.enrichment.scraper import ScrapeFailedError
from vramtop.enrichment.scrapers.sglang import SGLangScraper

SAMPLE_RESPONSE = {
    "model_path": "/models/meta-llama/Llama-3-8b",
    "tokenizer_path": "/models/meta-llama/Llama-3-8b",
    "is_generation": True,
    "max_total_num_tokens": 32768,
    "context_length": 8192,
}


class TestSGLangScraper:
    def test_parse_full_response(self) -> None:
        scraper = SGLangScraper()
        result = scraper._parse_response(json.dumps(SAMPLE_RESPONSE).encode())

        assert result["model_path"] == "/models/meta-llama/Llama-3-8b"
        assert result["tokenizer_path"] == "/models/meta-llama/Llama-3-8b"
        assert result["is_generation"] is True
        assert result["max_total_num_tokens"] == 32768
        assert result["context_length"] == 8192

    def test_parse_empty_response(self) -> None:
        scraper = SGLangScraper()
        result = scraper._parse_response(json.dumps({}).encode())
        assert result["model_path"] is None
        assert result["tokenizer_path"] is None
        assert result["is_generation"] is None
        assert result["max_total_num_tokens"] is None
        assert result["context_length"] is None

    def test_parse_partial_response(self) -> None:
        scraper = SGLangScraper()
        data = {"model_path": "/some/path", "context_length": 4096}
        result = scraper._parse_response(json.dumps(data).encode())
        assert result["model_path"] == "/some/path"
        assert result["context_length"] == 4096
        assert result["is_generation"] is None

    def test_parse_extra_fields_ignored(self) -> None:
        scraper = SGLangScraper()
        data = {
            "model_path": "/test",
            "extra_unknown_field": "should be ignored",
            "another_one": 42,
        }
        result = scraper._parse_response(json.dumps(data).encode())
        assert result["model_path"] == "/test"
        assert "extra_unknown_field" not in result

    def test_parse_invalid_json(self) -> None:
        scraper = SGLangScraper()
        with pytest.raises(ScrapeFailedError, match="Invalid JSON"):
            scraper._parse_response(b"{broken json")

    def test_endpoint_is_get_model_info(self) -> None:
        scraper = SGLangScraper()
        assert scraper.endpoint == "/get_model_info"
