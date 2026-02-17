"""Tests for Ollama scraper — JSON /api/ps parsing."""

from __future__ import annotations

import json

import pytest

from vramtop.enrichment.scraper import ScrapeFailedError
from vramtop.enrichment.scrapers.ollama import OllamaScraper

SAMPLE_RESPONSE = {
    "models": [
        {
            "name": "llama3:8b",
            "size": 4_700_000_000,
            "size_vram": 4_200_000_000,
            "details": {
                "family": "llama",
                "parameter_size": "8B",
                "quantization_level": "Q4_0",
            },
        },
        {
            "name": "mistral:7b",
            "size": 3_800_000_000,
            "size_vram": 3_500_000_000,
            "details": {
                "family": "mistral",
                "parameter_size": "7B",
                "quantization_level": "Q4_K_M",
            },
        },
    ]
}


class TestOllamaScraper:
    def test_parse_full_response(self) -> None:
        scraper = OllamaScraper()
        result = scraper._parse_response(json.dumps(SAMPLE_RESPONSE).encode())

        assert len(result["models"]) == 2
        m0 = result["models"][0]
        assert m0["name"] == "llama3:8b"
        assert m0["size"] == 4_700_000_000
        assert m0["size_vram"] == 4_200_000_000
        assert m0["family"] == "llama"
        assert m0["parameter_size"] == "8B"
        assert m0["quantization_level"] == "Q4_0"

    def test_parse_empty_models(self) -> None:
        scraper = OllamaScraper()
        result = scraper._parse_response(json.dumps({"models": []}).encode())
        assert result == {"models": []}

    def test_parse_no_models_key(self) -> None:
        scraper = OllamaScraper()
        result = scraper._parse_response(json.dumps({}).encode())
        assert result == {"models": []}

    def test_parse_extra_fields_ignored(self) -> None:
        scraper = OllamaScraper()
        data = {
            "models": [
                {
                    "name": "test:1b",
                    "size": 1000,
                    "size_vram": 800,
                    "details": {"family": "test", "extra_field": "ignored"},
                    "unknown_top_field": True,
                }
            ],
            "extra_root": 42,
        }
        result = scraper._parse_response(json.dumps(data).encode())
        assert result["models"][0]["name"] == "test:1b"

    def test_parse_invalid_json(self) -> None:
        scraper = OllamaScraper()
        with pytest.raises(ScrapeFailedError, match="Invalid JSON"):
            scraper._parse_response(b"not json at all")

    def test_parse_minimal_model(self) -> None:
        scraper = OllamaScraper()
        data = {"models": [{"name": "tiny:latest"}]}
        result = scraper._parse_response(json.dumps(data).encode())
        m = result["models"][0]
        assert m["name"] == "tiny:latest"
        assert m["size"] == 0
        assert m["size_vram"] == 0
        assert m["family"] is None

    def test_endpoint_is_api_ps(self) -> None:
        scraper = OllamaScraper()
        assert scraper.endpoint == "/api/ps"
