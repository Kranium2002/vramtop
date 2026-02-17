"""Ollama scraper — GET /api/ps JSON."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ConfigDict

from vramtop.enrichment.scraper import BaseScraper, ScrapeFailedError


class OllamaModelDetails(BaseModel):
    """Details sub-object in Ollama /api/ps response."""

    model_config = ConfigDict(extra="ignore")

    family: str | None = None
    parameter_size: str | None = None
    quantization_level: str | None = None


class OllamaModel(BaseModel):
    """A single running model from Ollama /api/ps."""

    model_config = ConfigDict(extra="ignore")

    name: str
    size: int = 0
    size_vram: int = 0
    details: OllamaModelDetails = OllamaModelDetails()


class OllamaResponse(BaseModel):
    """Root response from Ollama /api/ps."""

    model_config = ConfigDict(extra="ignore")

    models: list[OllamaModel] = []


class OllamaScraper(BaseScraper):
    """Scrape Ollama /api/ps endpoint."""

    endpoint: str = "/api/ps"

    def _parse_response(self, raw: bytes) -> dict[str, Any]:
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ScrapeFailedError("Invalid JSON from Ollama /api/ps") from exc

        try:
            resp = OllamaResponse.model_validate(data)
        except Exception as exc:
            raise ScrapeFailedError(f"Schema validation failed: {exc}") from exc

        models = []
        for m in resp.models:
            models.append({
                "name": m.name,
                "size": m.size,
                "size_vram": m.size_vram,
                "family": m.details.family,
                "parameter_size": m.details.parameter_size,
                "quantization_level": m.details.quantization_level,
            })

        return {"models": models}
