"""SGLang scraper — GET /get_model_info JSON."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ConfigDict

from vramtop.enrichment.scraper import BaseScraper, ScrapeFailedError


class SGLangModelInfo(BaseModel):
    """Response model for SGLang /get_model_info."""

    model_config = ConfigDict(extra="ignore")

    model_path: str | None = None
    tokenizer_path: str | None = None
    is_generation: bool | None = None
    max_total_num_tokens: int | None = None
    context_length: int | None = None


class SGLangScraper(BaseScraper):
    """Scrape SGLang /get_model_info endpoint."""

    endpoint: str = "/get_model_info"

    def _parse_response(self, raw: bytes) -> dict[str, Any]:
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ScrapeFailedError("Invalid JSON from SGLang /get_model_info") from exc

        try:
            info = SGLangModelInfo.model_validate(data)
        except Exception as exc:
            raise ScrapeFailedError(f"Schema validation failed: {exc}") from exc

        return {
            "model_path": info.model_path,
            "tokenizer_path": info.tokenizer_path,
            "is_generation": info.is_generation,
            "max_total_num_tokens": info.max_total_num_tokens,
            "context_length": info.context_length,
        }
