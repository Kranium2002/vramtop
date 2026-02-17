"""GPU card widget composing memory bar, timeline, process table, and alerts."""

from __future__ import annotations

from typing import TYPE_CHECKING

from textual.widgets import Static

from vramtop.analysis.oom_predictor import OOMPrediction
from vramtop.ui.widgets.alerts import OOMAlert
from vramtop.ui.widgets.memory_bar import MemoryBar
from vramtop.ui.widgets.process_table import ProcessTable
from vramtop.ui.widgets.timeline import Timeline

if TYPE_CHECKING:
    from textual.app import ComposeResult

    from vramtop.analysis.phase_detector import PhaseState
    from vramtop.analysis.survival import SurvivalPrediction
    from vramtop.backends.base import GPUDevice


def _format_mem_mb(n: int) -> str:
    """Format bytes as MB with thousands separators."""
    mb = n // (1024 * 1024)
    return f"{mb:,} MB"


class GPUCard(Static):
    """Container widget for a single GPU, composing header, memory bar,
    timeline, process table, and OOM alert."""

    DEFAULT_CSS = """
    GPUCard {
        height: auto;
        border: solid $primary;
        padding: 0 1;
        margin: 0 0 1 0;
    }
    """

    def __init__(
        self,
        gpu_index: int = 0,
        *,
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
        disabled: bool = False,
    ) -> None:
        super().__init__(name=name, id=id, classes=classes, disabled=disabled)
        self._gpu_index = gpu_index
        self._lost = False

    def compose(self) -> ComposeResult:
        yield Static("", id="gpu-header")
        yield MemoryBar()
        yield Timeline()
        yield ProcessTable()
        yield OOMAlert()

    def update_device(
        self,
        device: GPUDevice,
        phase_states: dict[int, PhaseState],
        oom_prediction: OOMPrediction | None,
        survival_states: dict[int, SurvivalPrediction] | None = None,
        enrichments: dict[int, dict[str, object]] | None = None,
    ) -> None:
        """Update all child widgets with new device data.

        Args:
            device: Current GPU device snapshot.
            phase_states: Mapping from PID to PhaseState.
            oom_prediction: Current OOM prediction or None.
            survival_states: Optional mapping from PID to SurvivalPrediction.
            enrichments: Optional mapping from PID to enrichment data dict.
        """
        if self._lost:
            return

        # Update header with formatted memory and Unicode separators
        header = self.query_one("#gpu-header", Static)
        used_str = _format_mem_mb(device.used_memory_bytes)
        total_str = _format_mem_mb(device.total_memory_bytes)

        # Color-coded temperature
        temp = device.temperature_celsius
        if temp >= 85:
            temp_str = f"[bold red]{temp}°C[/bold red]"
        elif temp >= 70:
            temp_str = f"[yellow]{temp}°C[/yellow]"
        else:
            temp_str = f"[green]{temp}°C[/green]"

        # Color-coded utilization
        util = device.gpu_util_percent
        if util >= 90:
            util_str = f"[bold red]{util}%[/bold red]"
        elif util >= 60:
            util_str = f"[yellow]{util}%[/yellow]"
        else:
            util_str = f"[green]{util}%[/green]"

        header.update(
            f"[bold]GPU {device.index}[/bold] {device.name}"
            f"  [dim]|[/dim]  {used_str}/{total_str}"
            f"  [dim]|[/dim]  Util {util_str}"
            f"  [dim]|[/dim]  {temp_str}"
            f"  [dim]|[/dim]  {device.power_watts:.0f}W"
        )

        # Update memory bar (used, free, total — reserved is derived)
        mem_bar = self.query_one(MemoryBar)
        mem_bar.update_memory(
            device.used_memory_bytes,
            device.free_memory_bytes,
            device.total_memory_bytes,
        )

        # Update timeline
        timeline = self.query_one(Timeline)
        timeline.add_value(
            float(device.used_memory_bytes),
            float(device.total_memory_bytes),
        )

        # Update process table
        proc_table = self.query_one(ProcessTable)
        proc_table.update_processes(
            list(device.processes),
            phase_states,
            device.total_memory_bytes,
            survival_states=survival_states,
            enrichments=enrichments,
        )

        # Update OOM alert
        oom_alert = self.query_one(OOMAlert)
        if oom_prediction is not None:
            oom_alert.update_prediction(oom_prediction)
        else:
            from vramtop.analysis.oom_predictor import Severity
            oom_alert.update_prediction(
                OOMPrediction(None, None, 0.0, Severity.NONE, "Stable")
            )

    def mark_lost(self) -> None:
        """Mark this GPU card as LOST (GPU fell off bus)."""
        self._lost = True
        header = self.query_one("#gpu-header", Static)
        header.update(
            f"[bold red]GPU {self._gpu_index}: LOST -- recovery needed[/bold red]"
        )
        # Hide dynamic widgets
        import contextlib

        with contextlib.suppress(Exception):
            self.query_one(MemoryBar).display = False
        with contextlib.suppress(Exception):
            self.query_one(Timeline).display = False
        with contextlib.suppress(Exception):
            self.query_one(ProcessTable).display = False
        with contextlib.suppress(Exception):
            self.query_one(OOMAlert).add_class("hidden")
