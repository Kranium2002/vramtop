"""Framework-specific HTTP scrapers — registry and discovery."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import psutil  # type: ignore[import-untyped]

if TYPE_CHECKING:
    from vramtop.config import ScrapingConfig
    from vramtop.enrichment.scraper import BaseScraper

logger = logging.getLogger(__name__)


def _build_registry() -> dict[str, type[BaseScraper]]:
    """Lazily import scraper classes to build the registry."""
    from vramtop.enrichment.scrapers.llamacpp import LlamaCppScraper
    from vramtop.enrichment.scrapers.ollama import OllamaScraper
    from vramtop.enrichment.scrapers.sglang import SGLangScraper
    from vramtop.enrichment.scrapers.vllm import VLLMScraper

    return {
        "vllm": VLLMScraper,
        "ollama": OllamaScraper,
        "sglang": SGLangScraper,
        "llamacpp": LlamaCppScraper,
    }


SCRAPER_REGISTRY: dict[str, type[BaseScraper]] | None = None


def _get_registry() -> dict[str, type[BaseScraper]]:
    global SCRAPER_REGISTRY  # noqa: PLW0603
    if SCRAPER_REGISTRY is None:
        SCRAPER_REGISTRY = _build_registry()
    return SCRAPER_REGISTRY


def get_scraper(framework: str, config: ScrapingConfig | None = None) -> BaseScraper | None:
    """Return an instantiated scraper for *framework*, or None."""
    registry = _get_registry()
    cls = registry.get(framework)
    if cls is None:
        return None
    return cls(config=config)


def detect_port(pid: int, framework: str) -> int | None:
    """Detect the listening port for a known framework process.

    Uses psutil to find LISTEN sockets owned by *pid*.
    Returns the first port found, or None.
    """
    # Default ports by framework
    default_ports: dict[str, int] = {
        "vllm": 8000,
        "ollama": 11434,
        "sglang": 8000,
        "llamacpp": 8080,
    }

    try:
        connections = psutil.net_connections(kind="inet")
    except (psutil.AccessDenied, OSError):
        return None

    pid_ports: list[int] = []
    for conn in connections:
        if (
            conn.status == "LISTEN"
            and conn.pid == pid
            and conn.laddr is not None
        ):
            pid_ports.append(conn.laddr.port)

    if not pid_ports:
        return None

    # Prefer the default port for this framework if the process is listening on it
    default = default_ports.get(framework)
    if default is not None and default in pid_ports:
        return default

    return pid_ports[0]
