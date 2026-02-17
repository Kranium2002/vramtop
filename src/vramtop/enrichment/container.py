"""Docker/Podman container detection.

Two modes:
- ``detect_container()``: module-level cache — detects if *vramtop itself* runs
  inside a container (checked once at startup).
- ``detect_process_container(pid)``: per-process detection — reads
  ``/proc/{pid}/cgroup`` to determine which container a *monitored* process
  belongs to.  Used by the enrichment pipeline so that host-side vramtop can
  identify the container of each GPU process.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

from vramtop.permissions import is_same_user

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


# cgroupv1: /docker/<id>, /podman/<id>, /containerd/<id>
_CGROUP_V1_RE = re.compile(
    r"(?:/docker/|/podman/|/containerd/)([a-f0-9]{12,64})"
)

# cgroupv2: docker-<id>.scope, podman-<id>.scope, cri-containerd-<id>.scope
_CGROUP_V2_RE = re.compile(
    r"(?:docker-|podman-|cri-containerd-)([a-f0-9]{12,64})\.scope"
)


def _parse_container_from_cgroup(content: str) -> ContainerInfo | None:
    """Parse container info from cgroup file content.

    Pure function — easy to test without filesystem access.
    Supports both cgroupv1 and cgroupv2 formats.

    Returns:
        ContainerInfo with runtime and short (12-char) container ID,
        or None if no container pattern is found.
    """
    # Try cgroupv1 first (more specific paths)
    match = _CGROUP_V1_RE.search(content)
    if match:
        full_match = match.group(0)
        cid = match.group(1)[:12]
        if "/docker/" in full_match:
            return ContainerInfo(runtime="docker", container_id=cid)
        if "/podman/" in full_match:
            return ContainerInfo(runtime="podman", container_id=cid)
        if "/containerd/" in full_match:
            return ContainerInfo(runtime="containerd", container_id=cid)

    # Try cgroupv2 (systemd scope names)
    match = _CGROUP_V2_RE.search(content)
    if match:
        full_match = match.group(0)
        cid = match.group(1)[:12]
        if full_match.startswith("docker-"):
            return ContainerInfo(runtime="docker", container_id=cid)
        if full_match.startswith("podman-"):
            return ContainerInfo(runtime="podman", container_id=cid)
        if full_match.startswith("cri-containerd-"):
            return ContainerInfo(runtime="containerd", container_id=cid)

    return None


def _parse_container_id_from_cgroup() -> str | None:
    """Extract container ID from /proc/1/cgroup."""
    try:
        with open("/proc/1/cgroup") as f:
            content = f.read(8192)
    except (FileNotFoundError, PermissionError):
        return None

    info = _parse_container_from_cgroup(content)
    return info.container_id if info else None


def detect_process_container(pid: int) -> ContainerInfo | None:
    """Detect which container a process belongs to by reading its cgroup.

    Reads ``/proc/{pid}/cgroup`` and parses container ID patterns.
    Only reads cgroups for same-UID processes (security boundary).

    Args:
        pid: Process ID to inspect.

    Returns:
        ContainerInfo if the process is in a container, else None.
    """
    if not is_same_user(pid):
        return None

    try:
        with open(f"/proc/{pid}/cgroup") as f:
            content = f.read(8192)
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return None

    return _parse_container_from_cgroup(content)


def _reset_cache() -> None:
    """Reset the detection cache (for testing)."""
    global _cached_result, _detection_done
    _cached_result = None
    _detection_done = False
