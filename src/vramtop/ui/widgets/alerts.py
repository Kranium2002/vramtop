"""OOM alert widget for displaying out-of-memory warnings."""

from __future__ import annotations

from typing import TYPE_CHECKING

from textual.widgets import Static

if TYPE_CHECKING:
    from vramtop.analysis.oom_predictor import OOMPrediction


class OOMAlert(Static):
    """Displays OOM warning when prediction is active."""

    DEFAULT_CSS = """
    OOMAlert {
        height: auto;
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
        self._active = False

    def update_prediction(self, prediction: OOMPrediction) -> None:
        """Update the alert with current OOM prediction."""
        from vramtop.analysis.oom_predictor import Severity

        if prediction.severity == Severity.CRITICAL:
            self._active = True
            self.remove_class("hidden")
            self.update(f"[bold red]!!! {prediction.display}[/bold red]")
        elif prediction.severity == Severity.WARNING:
            self._active = True
            self.remove_class("hidden")
            self.update(f"[bold yellow]! {prediction.display}[/bold yellow]")
        else:
            self._active = False
            self.add_class("hidden")
            self.update("")

    @property
    def active(self) -> bool:
        """Whether an OOM alert is currently displayed."""
        return self._active
