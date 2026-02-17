"""vLLM scraper — GET /metrics (Prometheus text format)."""

from __future__ import annotations

import re
from typing import Any

from vramtop.enrichment.scraper import BaseScraper, ScrapeFailedError

# Prometheus metric patterns (regex, not a library dependency)
_METRIC_RE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)"
    r"(?:\{[^}]*\})?\s+"
    r"(?P<value>[0-9eE.+\-]+|NaN|Inf|\+Inf|-Inf)$",
    re.MULTILINE,
)

# Metrics we care about
_WANTED = {
    "vllm:gpu_cache_usage_perc",
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:avg_generation_throughput_toks_per_s",
    "vllm:avg_prompt_throughput_toks_per_s",
}


class VLLMScraper(BaseScraper):
    """Scrape vLLM Prometheus /metrics endpoint."""

    endpoint: str = "/metrics"

    def _parse_response(self, raw: bytes) -> dict[str, Any]:
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ScrapeFailedError("Invalid UTF-8 in vLLM /metrics") from exc

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

            # Use short key names
            key = name.removeprefix("vllm:")
            metrics[key] = value

        return metrics
