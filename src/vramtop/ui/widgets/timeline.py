"""Timeline sparkline widget for memory usage history."""

from __future__ import annotations

from collections import deque

from textual.widgets import Static

_SPARK_CHARS = "\u2581\u2582\u2583\u2584\u2585\u2586\u2587\u2588"

# Color gradient for sparkline values (low → mid → high)
_SPARK_COLORS = [
    "#22c55e", "#22c55e",  # low: green
    "#4ade80", "#a3e635",  # low-mid: lime
    "#facc15", "#f59e0b",  # mid: yellow/amber
    "#fb923c", "#ef4444",  # high: orange/red
]


class Timeline(Static):
    """Horizontal sparkline showing memory usage history with color gradient."""

    DEFAULT_CSS = """
    Timeline {
        height: 1;
    }
    """

    def __init__(
        self,
        maxlen: int = 60,
        *,
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
        disabled: bool = False,
    ) -> None:
        super().__init__("", name=name, id=id, classes=classes, disabled=disabled)
        self._history: deque[float] = deque(maxlen=maxlen)
        self._max_values: deque[float] = deque(maxlen=maxlen)

    def add_value(self, value: float, max_value: float) -> None:
        """Add a data point and re-render the sparkline.

        Args:
            value: Current value (e.g., used memory bytes).
            max_value: Maximum possible value (e.g., total memory bytes).
        """
        self._history.append(value)
        self._max_values.append(max_value)
        self._render_sparkline()

    def _render_sparkline(self) -> None:
        if not self._history:
            self.update("")
            return

        # Use the current max_value for normalization
        max_val = self._max_values[-1] if self._max_values else 1.0
        if max_val <= 0:
            max_val = 1.0

        parts: list[str] = []
        num_chars = len(_SPARK_CHARS)
        num_colors = len(_SPARK_COLORS)
        for val in self._history:
            ratio = max(0.0, min(1.0, val / max_val))
            char_idx = int(ratio * (num_chars - 1))
            color_idx = int(ratio * (num_colors - 1))
            color = _SPARK_COLORS[color_idx]
            parts.append(f"[{color}]{_SPARK_CHARS[char_idx]}[/{color}]")

        self.update("".join(parts))

    @property
    def history(self) -> list[float]:
        """Return a copy of the history values."""
        return list(self._history)
