"""Slide-in detail panel for a selected GPU process."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich.text import Text
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
            Static("", id="detail-chart"),
            Static("", id="detail-segments"),
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
        container_runtime = enrichment.get("container_runtime")
        container_id = enrichment.get("container_id")
        if container_runtime:
            ctr_display = str(container_runtime)
            if container_id:
                ctr_display += f" ({container_id})"
            lines.append(f"[bold]Container:[/bold] {ctr_display}")

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

        # Deep mode: PyTorch internals
        scrape_data = enrichment.get("scrape_data")
        if isinstance(scrape_data, dict) and scrape_data.get("deep_mode") is True:
            allocated = scrape_data.get("allocated_mb")
            reserved = scrape_data.get("reserved_mb")
            active = scrape_data.get("active_mb")
            num_allocs = scrape_data.get("num_allocs")
            deep_segments = scrape_data.get("segments")

            lines.append("")
            lines.append("[bold]--- PyTorch Internals ---[/bold]")
            if isinstance(allocated, (int, float)):
                lines.append(
                    f"[bold]Allocated:[/bold] {allocated:.1f} MiB"
                    " [dim](tensor memory)[/dim]"
                )
            if isinstance(reserved, (int, float)):
                lines.append(
                    f"[bold]Reserved:[/bold] {reserved:.1f} MiB"
                    " [dim](allocator pool)[/dim]"
                )
            if isinstance(active, (int, float)):
                lines.append(f"[bold]Active:[/bold] {active:.1f} MiB [dim](in-use in pool)[/dim]")
            if (
                isinstance(reserved, (int, float))
                and isinstance(allocated, (int, float))
                and reserved > 0
            ):
                frag = (reserved - allocated) / reserved * 100
                lines.append(f"[bold]Fragmentation:[/bold] {frag:.1f}%")
            if isinstance(num_allocs, int):
                seg_str = ""
                if isinstance(deep_segments, int):
                    seg_str = f", Segments: {deep_segments}"
                lines.append(f"[bold]Allocs:[/bold] {num_allocs}{seg_str}")

        # Build PELT analysis section (chart + segments)
        chart_content, segments_content = self._build_pelt_section(enrichment, lines)

        # Reset title (clears any [EXITED] tag from a previous process)
        title = self.query_one("#detail-title", Static)
        title.update("[bold]Process Details[/bold]")

        body = self.query_one("#detail-body", Static)
        body.update("\n".join(lines))

        chart_widget = self.query_one("#detail-chart", Static)
        chart_widget.update(chart_content)

        seg_widget = self.query_one("#detail-segments", Static)
        seg_widget.update(segments_content)

        self.add_class("visible")

    def _build_pelt_section(
        self,
        enrichment: dict[str, object],
        lines: list[str],
    ) -> tuple[Text, Text]:
        """Build chart and segment summary for PELT analysis.

        Returns (chart_renderable, segments_renderable) for the Static widgets.
        Appends header lines to *lines* if PELT data is available.
        """
        empty = Text("")

        pelt_segments_raw = enrichment.get("pelt_segments")
        pelt_ts_raw = enrichment.get("pelt_timeseries")

        if isinstance(pelt_segments_raw, list) and pelt_segments_raw:
            from vramtop.analysis.segment_labels import (
                LABEL_COLORS,
                LABEL_DESCRIPTIONS,
                SegmentInfo,
            )
            from vramtop.ui.widgets.memory_chart import (
                render_memory_chart,
                render_segment_summary,
            )

            segments: list[SegmentInfo] = pelt_segments_raw
            ts: list[float] = pelt_ts_raw if isinstance(pelt_ts_raw, list) else []

            n_cps_raw = enrichment.get("pelt_changepoints")
            n_cps = len(n_cps_raw) if isinstance(n_cps_raw, list) else 0

            # GPU total for context
            vram_total_raw = enrichment.get("vram_total_bytes")
            gpu_total_mb = (
                float(vram_total_raw) / (1024 * 1024)
                if isinstance(vram_total_raw, (int, float))
                else 0.0
            )

            lines.append("")
            lines.append("[bold]--- Memory Timeline ---[/bold]")

            # Current state: label + description
            if segments:
                current_seg = segments[-1]
                color = LABEL_COLORS.get(current_seg.label, "white")
                desc = LABEL_DESCRIPTIONS.get(current_seg.label, "")
                lines.append(
                    f"[bold]Now:[/bold] "
                    f"[{color}]{current_seg.label.value}[/{color}]"
                )
                if desc:
                    lines.append(f"     [dim]{desc}[/dim]")

                # Peak memory and range
                if ts:
                    peak = max(ts)
                    current = ts[-1]
                    peak_str = self._fmt_mb(peak)
                    cur_str = self._fmt_mb(current)
                    lines.append(
                        f"[bold]Peak:[/bold] {peak_str}"
                        f"  [bold]Now:[/bold] {cur_str}"
                    )
                    if gpu_total_mb > 0:
                        peak_pct = peak / gpu_total_mb * 100
                        lines.append(
                            f"[dim]Peak used {peak_pct:.0f}% "
                            f"of GPU memory[/dim]"
                        )

            lines.append(
                f"[dim]{n_cps + 1} phases detected, "
                f"{len(ts)} samples[/dim]"
            )
            lines.append("")

            # Render chart
            chart = render_memory_chart(ts, segments, width=44, height=8)
            # Render segment summary with GPU context
            seg_summary = render_segment_summary(
                segments, gpu_total_mb=gpu_total_mb,
            )

            return chart, seg_summary

        # Fallback: text-only PELT if segments not available but phases are
        pelt_phases_raw = enrichment.get("pelt_phases")
        if isinstance(pelt_phases_raw, list) and pelt_phases_raw:
            pelt_phases_list: list[str] = [str(p) for p in pelt_phases_raw]
            pelt_cps_raw = enrichment.get("pelt_changepoints")
            n_cps = len(pelt_cps_raw) if isinstance(pelt_cps_raw, list) else 0
            pelt_current = str(enrichment.get("pelt_current_phase", "unknown"))
            n_samples = enrichment.get("pelt_num_samples", 0)

            lines.append("")
            lines.append("[bold]--- Memory Timeline ---[/bold]")
            lines.append(
                f"[bold]Now:[/bold] {pelt_current}"
            )

            # Segment timeline as text fallback
            phase_map: dict[str, str] = {
                "stable": "[green]stable[/green]",
                "growing": "[yellow]growing[/yellow]",
                "shrinking": "[cyan]shrinking[/cyan]",
                "volatile": "[red]volatile[/red]",
            }
            seg_labels: list[str] = [
                phase_map.get(p, p) for p in pelt_phases_list
            ]
            if len(seg_labels) <= 6:
                lines.append(
                    "[bold]Timeline:[/bold] "
                    + " -> ".join(seg_labels)
                )
            else:
                lines.append(
                    "[bold]Timeline:[/bold] "
                    + " -> ".join(seg_labels[:3])
                    + " -> ... -> "
                    + " -> ".join(seg_labels[-2:])
                )
            lines.append(
                f"[dim]{n_cps + 1} phases, {n_samples} samples[/dim]"
            )

        return empty, empty

    @staticmethod
    def _fmt_mb(mb: float) -> str:
        """Format MB with appropriate unit."""
        if mb >= 1024:
            return f"{mb / 1024:.1f} GB"
        return f"{mb:.0f} MB"

    def mark_exited(self) -> None:
        """Mark the current process as exited without clearing data.

        Keeps all enrichment/deep mode data visible for post-mortem
        inspection, but adds an [EXITED] banner so the user knows.
        """
        if self._pid is None:
            return
        title = self.query_one("#detail-title", Static)
        title.update(
            "[bold]Process Details[/bold]  [bold red]\\[EXITED][/bold red]"
        )

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
