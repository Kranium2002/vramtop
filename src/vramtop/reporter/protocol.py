"""Wire protocol for deep mode Unix socket IPC."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass

logger = logging.getLogger(__name__)

PROTOCOL_VERSION = 1


@dataclass(frozen=True, slots=True)
class HandshakeMsg:
    """Initial handshake sent by reporter to identify itself."""

    v: int
    pid: int
    framework: str
    cuda_device: int


@dataclass(frozen=True, slots=True)
class MemoryMsg:
    """Periodic memory stats sent by reporter."""

    ts: float
    allocated_mb: float
    reserved_mb: float
    active_mb: float
    num_allocs: int
    segments: int


def to_json(msg: HandshakeMsg | MemoryMsg) -> str:
    """Serialize a message to a single-line JSON string."""
    d = asdict(msg)
    if isinstance(msg, HandshakeMsg):
        d["type"] = "handshake"
    else:
        d["type"] = "memory"
    return json.dumps(d, separators=(",", ":"))


def parse_message(line: str) -> HandshakeMsg | MemoryMsg | None:
    """Parse a JSON line into a message, or None on failure."""
    try:
        d = json.loads(line)
    except (json.JSONDecodeError, TypeError):
        return None

    if not isinstance(d, dict):
        return None

    msg_type = d.get("type")
    try:
        if msg_type == "handshake":
            return HandshakeMsg(
                v=int(d["v"]),
                pid=int(d["pid"]),
                framework=str(d["framework"]),
                cuda_device=int(d["cuda_device"]),
            )
        if msg_type == "memory":
            return MemoryMsg(
                ts=float(d["ts"]),
                allocated_mb=float(d["allocated_mb"]),
                reserved_mb=float(d["reserved_mb"]),
                active_mb=float(d["active_mb"]),
                num_allocs=int(d["num_allocs"]),
                segments=int(d["segments"]),
            )
    except (KeyError, ValueError, TypeError):
        return None

    return None
