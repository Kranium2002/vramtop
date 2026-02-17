"""EMA-based allocation rate tracking per GPU/process."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vramtop.backends.base import ProcessIdentity


@dataclass
class _RateEntry:
    ema_rate: float = 0.0
    last_update: float = 0.0


class AllocationRateTracker:
    """Track per-process allocation rates using exponential moving average.

    Keyed by (gpu_index, ProcessIdentity) tuples. Stale entries are pruned
    based on a configurable TTL.
    """

    def __init__(
        self,
        ema_alpha: float = 0.3,
        stale_ttl_seconds: float = 30.0,
    ) -> None:
        self.ema_alpha = ema_alpha
        self.stale_ttl = stale_ttl_seconds
        self._entries: dict[tuple[int, ProcessIdentity], _RateEntry] = {}

    def update(
        self,
        gpu_index: int,
        identity: ProcessIdentity,
        rate_mb_per_sec: float,
    ) -> float:
        """Update the EMA rate for a process and return the smoothed rate."""
        key = (gpu_index, identity)
        entry = self._entries.get(key)
        now = time.monotonic()

        if entry is None:
            entry = _RateEntry(ema_rate=rate_mb_per_sec, last_update=now)
            self._entries[key] = entry
            return rate_mb_per_sec

        entry.ema_rate = (
            self.ema_alpha * rate_mb_per_sec + (1 - self.ema_alpha) * entry.ema_rate
        )
        entry.last_update = now
        return entry.ema_rate

    def get_rate(
        self, gpu_index: int, identity: ProcessIdentity
    ) -> float | None:
        """Return the current EMA rate, or None if not tracked."""
        entry = self._entries.get((gpu_index, identity))
        if entry is None:
            return None
        return entry.ema_rate

    def prune_stale(self) -> int:
        """Remove entries not updated within the TTL. Returns count removed."""
        now = time.monotonic()
        stale_keys = [
            key
            for key, entry in self._entries.items()
            if (now - entry.last_update) > self.stale_ttl
        ]
        for key in stale_keys:
            del self._entries[key]
        return len(stale_keys)

    def __len__(self) -> int:
        return len(self._entries)
