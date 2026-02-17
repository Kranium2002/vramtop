"""Slide-in detail panel for a selected GPU process."""

from __future__ import annotations

from typing import TYPE_CHECKING

from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.widget import Widget
from textual.widgets import Static

if TYPE_CHECKING:
    from textual.app import ComposeResult


class DetailPanel(Widget):
    """Side panel showing detailed info for a selected GPU process.

    Triggered by Enter or 'd' on a selected process row.
    """

    DEFAULT_CSS = """
    DetailPanel {
        dock: right;
        width: 50;
        height: 100%;
        border-left: solid $primary;
        padding: 1 2;
        background: $surface;
        display: none;
    }
    DetailPanel.visible {
        display: block;
    }
    DetailPanel #detail-title {
        text-style: bold;
        margin: 0 0 1 0;
    }
    DetailPanel .detail-section {
        margin: 0 0 1 0;
    }
    DetailPanel .detail-label {
        text-style: bold;
        color: $text-muted;
    }
    """

    BINDINGS = [
        Binding("escape", "close", "Close", show=True),
        Binding("k", "open_kill", "Kill process", show=True),
    ]

    def __init__(
        self,
        *,
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
    ) -> None:
        super().__init__(name=name, id=id, classes=classes)
        self._pid: int | None = None
        self._process_name: str = ""
        self._enrichment: dict[str, object] = {}

    def compose(self) -> ComposeResult:
        yield Static("Process Details", id="detail-title")
        yield VerticalScroll(
            Static("", id="detail-body"),
            id="detail-scroll",
        )

    def show_process(self, pid: int, name: str, enrichment: dict[str, object]) -> None:
        """Display details for a process.

        Args:
            pid: Process ID.
            name: Process name (sanitized).
            enrichment: Dict of enrichment data (flexible schema).
        """
        self._pid = pid
        self._process_name = name
        self._enrichment = enrichment

        lines: list[str] = []
        lines.append(f"[bold]PID:[/bold] {pid}")
        lines.append(f"[bold]Command:[/bold] {name}")

        # Framework info
        framework = enrichment.get("framework")
        if framework:
            lines.append(f"[bold]Framework:[/bold] {framework}")

        # Container info
        container = enrichment.get("container")
        if container:
            lines.append(f"[bold]Container:[/bold] {container}")

        # MPS status
        mps = enrichment.get("mps")
        if mps:
            lines.append(f"[bold]MPS:[/bold] {mps}")

        lines.append("")

        # VRAM breakdown
        vram_used = enrichment.get("vram_used_bytes")
        vram_total = enrichment.get("vram_total_bytes")
        if isinstance(vram_used, (int, float)):
            used_mb = vram_used / (1024 * 1024)
            lines.append(f"[bold]VRAM Used:[/bold] {used_mb:.0f} MiB")
        if isinstance(vram_total, (int, float)):
            total_mb = vram_total / (1024 * 1024)
            lines.append(f"[bold]VRAM Total:[/bold] {total_mb:.0f} MiB")

        weight_est = enrichment.get("weight_estimate_mb")
        if weight_est is not None:
            lines.append(f"[bold]Weight Est:[/bold] ~{weight_est:.0f} MiB [dim](~estimate)[/dim]")

        dynamic_est = enrichment.get("dynamic_estimate_mb")
        if dynamic_est is not None:
            lines.append(f"[bold]Dynamic Est:[/bold] ~{dynamic_est:.0f} MiB [dim](~estimate)[/dim]")

        lines.append("")

        # Phase / rate / confidence
        phase = enrichment.get("phase")
        if phase:
            lines.append(f"[bold]Phase:[/bold] {phase}")
        rate = enrichment.get("rate_mb_per_sec")
        if rate is not None:
            lines.append(f"[bold]Rate:[/bold] {rate:+.1f} MB/s")
        confidence = enrichment.get("confidence")
        if isinstance(confidence, (int, float)):
            pct = confidence * 100
            lines.append(f"[bold]Confidence:[/bold] {pct:.0f}%")

        # OOM risk
        oom = enrichment.get("oom_display")
        if oom and oom != "Stable":
            lines.append(f"\n[bold red]OOM Risk:[/bold red] {oom}")

        body = self.query_one("#detail-body", Static)
        body.update("\n".join(lines))

        self.add_class("visible")

    def action_close(self) -> None:
        """Close the detail panel."""
        self.remove_class("visible")
        self._pid = None

    def action_open_kill(self) -> None:
        """Request the app to open the kill dialog for the current process."""
        if self._pid is not None:
            self.app.open_kill_dialog(self._pid, self._process_name)  # type: ignore[attr-defined]

    @property
    def current_pid(self) -> int | None:
        return self._pid
