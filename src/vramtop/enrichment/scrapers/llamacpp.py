"""llama.cpp scraper — GET /health + GET /metrics (Prometheus)."""

from __future__ import annotations

import re
from typing import Any

from vramtop.enrichment.scraper import BaseScraper, ScrapeFailedError

# Prometheus metric regex (same approach as vLLM scraper)
_METRIC_RE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)"
    r"(?:\{[^}]*\})?\s+"
    r"(?P<value>[0-9eE.+\-]+|NaN|Inf|\+Inf|-Inf)$",
    re.MULTILINE,
)

# Metrics we extract from llama.cpp /metrics
_WANTED = {
    "llamacpp_kv_cache_usage",
    "llamacpp_requests_processing",
    "llamacpp_requests_pending",
    "llamacpp_tokens_predicted_total",
}


class LlamaCppScraper(BaseScraper):
    """Scrape llama.cpp /health + /metrics endpoints.

    Primary endpoint is /metrics (Prometheus). The /health endpoint
    provides a simple status check.
    """

    endpoint: str = "/metrics"

    def _parse_response(self, raw: bytes) -> dict[str, Any]:
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ScrapeFailedError("Invalid UTF-8 in llama.cpp /metrics") from exc

        metrics: dict[str, Any] = {}

        for match in _METRIC_RE.finditer(text):
            name = match.group("name")
            value_str = match.group("value")

            if name not in _WANTED:
                continue

            try:
                value = float(value_str)
            except ValueError:
                continue

            # Strip prefix for cleaner keys
            key = name.removeprefix("llamacpp_")
            metrics[key] = value

        return metrics
