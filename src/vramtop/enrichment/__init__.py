"""Process enrichment: framework detection, model files, containers."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from vramtop.permissions import is_same_user

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ModelFileInfo:
    """A model file opened by a GPU process."""

    path: str
    size_bytes: int
    extension: str


@dataclass(slots=True)
class EnrichmentResult:
    """Aggregated enrichment data for a single GPU process."""

    framework: str | None = None
    framework_version: str | None = None
    model_files: list[ModelFileInfo] = field(default_factory=list)
    estimated_model_size_bytes: int | None = None
    container_runtime: str | None = None
    container_id: str | None = None
    is_mps_client: bool = False
    cmdline: str | None = None
    scrape_data: dict[str, Any] | None = None


def enrich_process(
    pid: int, starttime: int, *, scraping_config: Any | None = None
) -> EnrichmentResult:
    """Orchestrate all enrichment layers for a process.

    Checks same-UID first. Each enricher runs in try/except so
    a single failure never blocks others.
    """
    result = EnrichmentResult()

    if not is_same_user(pid):
        return result

    # Framework detection
    try:
        from vramtop.enrichment.detector import detect_framework

        fw_name, fw_ver = detect_framework(pid)
        result.framework = fw_name
        result.framework_version = fw_ver
    except Exception:
        logger.debug("Framework detection failed for pid=%d", pid, exc_info=True)

    # Model file scanning
    try:
        from vramtop.enrichment.model_files import scan_model_files

        result.model_files = scan_model_files(pid)
        if result.model_files:
            result.estimated_model_size_bytes = sum(
                f.size_bytes for f in result.model_files
            )
    except Exception:
        logger.debug("Model file scan failed for pid=%d", pid, exc_info=True)

    # Container detection (cached, no per-process cost)
    try:
        from vramtop.enrichment.container import detect_container

        info = detect_container()
        if info is not None:
            result.container_runtime = info.runtime
            result.container_id = info.container_id
    except Exception:
        logger.debug("Container detection failed", exc_info=True)

    # MPS detection
    try:
        from vramtop.enrichment.mps import is_mps_client

        result.is_mps_client = is_mps_client(pid)
    except Exception:
        logger.debug("MPS detection failed for pid=%d", pid, exc_info=True)

    # Command line (for display)
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            raw = f.read(4096)
        result.cmdline = raw.replace(b"\x00", b" ").decode("utf-8", errors="replace").strip()
    except Exception:
        logger.debug("Cmdline read failed for pid=%d", pid, exc_info=True)

    # HTTP scraping (Layer 2 — opt-in, config-gated)
    if scraping_config is not None and scraping_config.enable and result.framework:
        try:
            from vramtop.enrichment.scrapers import detect_port, get_scraper

            scraper = get_scraper(result.framework, config=scraping_config)
            if scraper is not None:
                port = detect_port(pid, result.framework)
                if port is not None:
                    result.scrape_data = scraper.scrape(pid, port)
        except Exception:
            logger.debug(
                "Scraping failed for pid=%d framework=%s",
                pid,
                result.framework,
                exc_info=True,
            )

    return result
