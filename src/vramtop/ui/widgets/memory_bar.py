"""Horizontal memory bar widget showing used/reserved/free VRAM."""

from __future__ import annotations

from textual.widgets import Static


def _format_bytes(n: int) -> str:
    """Format byte count to human-readable string."""
    if n >= 1 << 30:
        return f"{n / (1 << 30):.1f} GiB"
    if n >= 1 << 20:
        return f"{n / (1 << 20):.0f} MiB"
    return f"{n} B"


class MemoryBar(Static):
    """Horizontal bar showing used/reserved/free VRAM with percentage.

    The bar has three segments:
    - **used** (green/yellow/red) — application memory
    - **reserved** (dim) — driver overhead (total - used - free)
    - **free** (empty) — allocatable
    """

    DEFAULT_CSS = """
    MemoryBar {
        height: 1;
    }
    """

    def __init__(
        self,
        *,
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
        disabled: bool = False,
    ) -> None:
        super().__init__("", name=name, id=id, classes=classes, disabled=disabled)
        self._used: int = 0
        self._free: int = 0
        self._total: int = 1

    def update_memory(
        self, used_bytes: int, free_bytes: int, total_bytes: int
    ) -> None:
        """Update the bar with current memory values."""
        self._used = used_bytes
        self._free = free_bytes
        self._total = max(total_bytes, 1)
        self._render_bar()

    def _render_bar(self) -> None:
        reserved = max(0, self._total - self._used - self._free)
        # Percentage of total that is truly unavailable (used + reserved)
        used_pct = self._used / self._total * 100
        reserved_pct = reserved / self._total * 100

        used_str = _format_bytes(self._used)
        total_str = _format_bytes(self._total)

        # Choose color for used segment based on utilization
        total_occupied_pct = used_pct + reserved_pct
        if total_occupied_pct < 70:
            color = "green"
        elif total_occupied_pct < 90:
            color = "yellow"
        else:
            color = "red"

        # Build a text-based bar (width 30 chars)
        bar_width = 30
        filled_used = int(bar_width * used_pct / 100)
        filled_rsv = int(bar_width * reserved_pct / 100)
        empty = bar_width - filled_used - filled_rsv

        bar_used = f"[{color}]{'█' * filled_used}[/{color}]"
        bar_rsv = f"[dim]{'▒' * filled_rsv}[/dim]" if filled_rsv > 0 else ""
        bar_free = "░" * empty

        label = f" {used_str}/{total_str} ({used_pct:.0f}%)"
        rsv_label = f" +{_format_bytes(reserved)} rsv" if reserved > 0 else ""

        self.update(f"Mem: {bar_used}{bar_rsv}{bar_free}{label}{rsv_label}")
