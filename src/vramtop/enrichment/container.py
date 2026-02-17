"""Docker/Podman container detection."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

# Module-level cache: detect once at startup
_cached_result: ContainerInfo | None = None
_detection_done: bool = False


@dataclass(frozen=True, slots=True)
class ContainerInfo:
    """Container runtime information."""

    runtime: str
    container_id: str | None


def detect_container() -> ContainerInfo | None:
    """Detect if running inside a container.

    Checks for Docker (/.dockerenv), Podman (/run/.containerenv),
    and cgroup-based detection. Result is cached after first call.
    """
    global _cached_result, _detection_done

    if _detection_done:
        return _cached_result

    _cached_result = _detect_container_impl()
    _detection_done = True
    return _cached_result


def get_nvidia_visible_devices() -> str | None:
    """Read NVIDIA_VISIBLE_DEVICES from own environment."""
    return os.environ.get("NVIDIA_VISIBLE_DEVICES")


def _detect_container_impl() -> ContainerInfo | None:
    """Internal detection logic."""
    # Docker: /.dockerenv file exists
    if os.path.exists("/.dockerenv"):
        cid = _parse_container_id_from_cgroup()
        return ContainerInfo(runtime="docker", container_id=cid)

    # Podman: /run/.containerenv exists
    if os.path.exists("/run/.containerenv"):
        cid = _parse_container_id_from_cgroup()
        return ContainerInfo(runtime="podman", container_id=cid)

    # Fallback: cgroup parsing
    cid = _parse_container_id_from_cgroup()
    if cid is not None:
        return ContainerInfo(runtime="unknown", container_id=cid)

    return None


_CGROUP_CONTAINER_RE = re.compile(
    r"(?:/docker/|/podman/|/containerd/)([a-f0-9]{12,64})"
)


def _parse_container_id_from_cgroup() -> str | None:
    """Extract container ID from /proc/1/cgroup."""
    try:
        with open("/proc/1/cgroup") as f:
            content = f.read(8192)
    except (FileNotFoundError, PermissionError):
        return None

    match = _CGROUP_CONTAINER_RE.search(content)
    if match:
        return match.group(1)[:12]  # Short container ID
    return None


def _reset_cache() -> None:
    """Reset the detection cache (for testing)."""
    global _cached_result, _detection_done
    _cached_result = None
    _detection_done = False
