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


# Gradient from green → yellow → red for memory pressure
_GRADIENT = [
    "#22c55e", "#4ade80", "#86efac",  # green
    "#a3e635", "#d9f99d",              # lime
    "#facc15", "#fbbf24", "#f59e0b",  # yellow/amber
    "#fb923c", "#f97316",              # orange
    "#ef4444", "#dc2626",              # red
]


def _gradient_color(pct: float) -> str:
    """Pick a color from the gradient based on 0-100 percentage."""
    idx = int(pct / 100 * (len(_GRADIENT) - 1))
    idx = max(0, min(len(_GRADIENT) - 1, idx))
    return _GRADIENT[idx]


class MemoryBar(Static):
    """Horizontal bar showing used/reserved/free VRAM with percentage.

    The bar has three segments:
    - **used** (gradient green→red) — application memory
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
        used_pct = self._used / self._total * 100
        reserved_pct = reserved / self._total * 100

        used_str = _format_bytes(self._used)
        total_str = _format_bytes(self._total)
        free_str = _format_bytes(self._free)

        # Pick color based on total pressure (used + reserved)
        total_occupied_pct = used_pct + reserved_pct
        color = _gradient_color(total_occupied_pct)

        # Build a text-based bar (width 30 chars)
        bar_width = 30
        filled_used = int(bar_width * used_pct / 100)
        filled_rsv = int(bar_width * reserved_pct / 100)
        empty = bar_width - filled_used - filled_rsv

        bar_used = f"[{color}]{'━' * filled_used}[/{color}]"
        bar_rsv = f"[dim]{'╌' * filled_rsv}[/dim]" if filled_rsv > 0 else ""
        bar_free = f"[dim]{'─' * empty}[/dim]"

        label = f" {used_str}/{total_str}"
        free_label = f" ({free_str} free)"
        rsv_label = f" [dim]+{_format_bytes(reserved)} rsv[/dim]" if reserved > 0 else ""

        self.update(f"├{bar_used}{bar_rsv}{bar_free}┤{label}{free_label}{rsv_label}")
