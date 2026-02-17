"""Process enrichment: framework detection, model files, containers."""

from __future__ import annotations

import logging
import os
import time as _time
from dataclasses import dataclass, field
from typing import Any

from vramtop.permissions import is_same_user

logger = logging.getLogger(__name__)

# --- Docker PID namespace resolution ---
# NVML reports host PIDs inside containers.  These PIDs don't exist in the
# container's /proc/, so is_same_user() fails and enrichment is skipped.
# We resolve this by scanning /proc/ for GPU-using processes and mapping
# NVML (host) PIDs to container PIDs.

_container_gpu_pids: list[int] = []
_container_gpu_pids_time: float = 0.0
_CONTAINER_GPU_SCAN_TTL = 30.0  # seconds


def _scan_container_gpu_pids() -> list[int]:
    """Find PIDs in our namespace that have /dev/nvidia* file descriptors.

    Excludes the current process (vramtop itself).
    """
    global _container_gpu_pids, _container_gpu_pids_time  # noqa: PLW0603

    now = _time.monotonic()
    if _container_gpu_pids and (now - _container_gpu_pids_time) < _CONTAINER_GPU_SCAN_TTL:
        return _container_gpu_pids

    my_uid = os.getuid()
    my_pid = os.getpid()
    result: list[int] = []

    try:
        entries = os.listdir("/proc")
    except OSError:
        return []

    for entry in entries:
        if not entry.isdigit():
            continue
        pid = int(entry)
        if pid == my_pid:
            continue
        try:
            if os.stat(f"/proc/{pid}").st_uid != my_uid:
                continue
            fd_dir = f"/proc/{pid}/fd"
            for fd_name in os.listdir(fd_dir):
                try:
                    target = os.readlink(f"{fd_dir}/{fd_name}")
                    if "/dev/nvidia" in target:
                        result.append(pid)
                        break
                except (FileNotFoundError, PermissionError, OSError):
                    continue
        except (FileNotFoundError, PermissionError, OSError):
            continue

    _container_gpu_pids = result
    _container_gpu_pids_time = now
    return result


def _resolve_pid(nvml_pid: int) -> int:
    """Resolve an NVML PID to a container PID if needed.

    If the PID exists in /proc/ (non-Docker or same namespace), returns
    it unchanged.  Otherwise scans for GPU-using container PIDs and
    returns the best match.
    """
    if os.path.exists(f"/proc/{nvml_pid}"):
        return nvml_pid

    # Docker PID namespace: NVML host PID not in our /proc/.
    gpu_pids = _scan_container_gpu_pids()
    if not gpu_pids:
        return nvml_pid  # Can't resolve; caller will handle gracefully

    # Single GPU process: unambiguous match.
    if len(gpu_pids) == 1:
        return gpu_pids[0]

    # Multiple GPU processes: return first non-vramtop process as best
    # effort.  Proper multi-process matching would need VRAM correlation.
    for pid in gpu_pids:
        try:
            with open(f"/proc/{pid}/comm", "rb") as f:
                comm = f.read(256).decode("utf-8", errors="replace").strip()
            if "vramtop" not in comm:
                return pid
        except (FileNotFoundError, PermissionError, OSError):
            continue

    return gpu_pids[0]


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
    pid: int, starttime: int = 0, *, scraping_config: Any | None = None
) -> EnrichmentResult:
    """Orchestrate all enrichment layers for a process.

    Checks same-UID first. Each enricher runs in try/except so
    a single failure never blocks others.

    Handles Docker PID namespace: NVML may report host PIDs that
    don't exist in the container's /proc/.  We resolve to the
    container PID for /proc reads.
    """
    result = EnrichmentResult()

    # Resolve Docker PID namespace (host PID → container PID).
    resolved_pid = _resolve_pid(pid)

    if not is_same_user(resolved_pid):
        return result

    # Use resolved_pid for all /proc reads (handles Docker PID namespace).
    # Keep original pid for NVML-level operations and deep mode socket scan.

    # Framework detection
    try:
        from vramtop.enrichment.detector import detect_framework

        fw_name, fw_ver = detect_framework(resolved_pid, starttime)
        result.framework = fw_name
        result.framework_version = fw_ver
    except Exception:
        logger.debug("Framework detection failed for pid=%d", resolved_pid, exc_info=True)

    # Model file scanning
    try:
        from vramtop.enrichment.model_files import scan_model_files

        result.model_files = scan_model_files(resolved_pid)
        if result.model_files:
            result.estimated_model_size_bytes = sum(
                f.size_bytes for f in result.model_files
            )
    except Exception:
        logger.debug("Model file scan failed for pid=%d", resolved_pid, exc_info=True)

    # Per-process container detection (reads /proc/{pid}/cgroup)
    try:
        from vramtop.enrichment.container import detect_process_container

        info = detect_process_container(resolved_pid)
        if info is not None:
            result.container_runtime = info.runtime
            result.container_id = info.container_id
    except Exception:
        logger.debug("Container detection failed for pid=%d", resolved_pid, exc_info=True)

    # MPS detection
    try:
        from vramtop.enrichment.mps import is_mps_client

        result.is_mps_client = is_mps_client(resolved_pid)
    except Exception:
        logger.debug("MPS detection failed for pid=%d", resolved_pid, exc_info=True)

    # Command line (for display)
    try:
        with open(f"/proc/{resolved_pid}/cmdline", "rb") as f:
            raw = f.read(4096)
        result.cmdline = raw.replace(b"\x00", b" ").decode("utf-8", errors="replace").strip()
    except Exception:
        logger.debug("Cmdline read failed for pid=%d", resolved_pid, exc_info=True)

    # HTTP scraping (Layer 2 — opt-in, config-gated)
    if scraping_config is not None and scraping_config.enable and result.framework:
        try:
            from vramtop.enrichment.scrapers import detect_port, get_scraper

            scraper = get_scraper(result.framework, config=scraping_config)
            if scraper is not None:
                port = detect_port(resolved_pid, result.framework)
                if port is not None:
                    result.scrape_data = scraper.scrape(resolved_pid, port)
        except Exception:
            logger.debug(
                "Scraping failed for pid=%d framework=%s",
                resolved_pid,
                result.framework,
                exc_info=True,
            )

    # Deep mode enrichment (Unix socket IPC).
    # Uses original pid first (exact match), then falls back to scanning
    # all sockets (handles Docker PID namespace).
    try:
        from vramtop.enrichment.deep_mode import get_deep_enrichment

        deep = get_deep_enrichment(pid)
        if deep is not None:
            result.scrape_data = result.scrape_data or {}
            result.scrape_data.update(deep)
    except Exception:
        logger.debug("Deep mode enrichment failed for pid=%d", pid, exc_info=True)

    return result
