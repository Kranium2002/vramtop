"""Phase badge widget showing memory allocation phase."""

from __future__ import annotations

import os

from textual.widgets import Static

from vramtop.analysis.phase_detector import Phase, PhaseState

_PHASE_SYMBOLS: dict[Phase, str] = {
    Phase.STABLE: "\u25ac STABLE",
    Phase.GROWING: "\u25b2 GROWING",
    Phase.SHRINKING: "\u25bc SHRINKING",
    Phase.VOLATILE: "\u301c VOLATILE",
}

_PHASE_TEXT: dict[Phase, str] = {
    Phase.STABLE: "[STABLE]",
    Phase.GROWING: "[GROWING]",
    Phase.SHRINKING: "[SHRINKING]",
    Phase.VOLATILE: "[VOLATILE]",
}

_PHASE_ACCESSIBLE: dict[Phase, str] = {
    Phase.STABLE: "\u25ac [STABLE] - Memory usage is steady",
    Phase.GROWING: "\u25b2 [GROWING] - Memory usage is increasing",
    Phase.SHRINKING: "\u25bc [SHRINKING] - Memory usage is decreasing",
    Phase.VOLATILE: "\u301c [VOLATILE] - Memory usage is fluctuating",
}

_PHASE_COLORS: dict[Phase, str] = {
    Phase.STABLE: "green",
    Phase.GROWING: "yellow",
    Phase.SHRINKING: "blue",
    Phase.VOLATILE: "red",
}


class PhaseBadge(Static):
    """Displays the current memory allocation phase with a symbol."""

    def __init__(
        self,
        *,
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
        disabled: bool = False,
    ) -> None:
        super().__init__("", name=name, id=id, classes=classes, disabled=disabled)

    def update_phase(self, phase_state: PhaseState) -> None:
        """Update the badge with the current phase state."""
        no_color = os.environ.get("NO_COLOR") is not None
        phase = phase_state.phase

        # Check if the app is in accessible mode
        accessible = False
        try:
            app = self.app
            if hasattr(app, "accessible"):
                accessible = getattr(app, "accessible", False)
        except Exception:
            pass

        if accessible:
            text = _PHASE_ACCESSIBLE[phase]
        elif no_color:
            text = _PHASE_TEXT[phase]
        else:
            symbol = _PHASE_SYMBOLS[phase]
            color = _PHASE_COLORS[phase]
            text = f"[{color}]{symbol}[/{color}]"

        self.update(text)
