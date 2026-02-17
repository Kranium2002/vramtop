"""Process table widget showing per-process VRAM usage."""

from __future__ import annotations

from typing import TYPE_CHECKING

from textual.widgets import DataTable

from vramtop.analysis.phase_detector import Phase, PhaseState
from vramtop.sanitize import sanitize_process_name

if TYPE_CHECKING:
    from vramtop.backends.base import GPUProcess

_PHASE_LABELS: dict[Phase, str] = {
    Phase.STABLE: "stable",
    Phase.GROWING: "growing",
    Phase.SHRINKING: "shrinking",
    Phase.VOLATILE: "volatile",
}

# Maximum display length for process names
_MAX_NAME_LEN = 40


def _format_bytes(n: int) -> str:
    """Format byte count to human-readable string with thousands separators."""
    mb = n // (1024 * 1024)
    if mb >= 1024:
        return f"{mb:,} MB"
    return f"{mb:,} MB"


def _truncate(name: str, max_len: int = _MAX_NAME_LEN) -> str:
    """Truncate a string with '...' if too long."""
    if len(name) <= max_len:
        return name
    return name[: max_len - 3] + "..."


class ProcessTable(DataTable[str]):
    """Table showing per-process GPU memory usage."""

    DEFAULT_CSS = """
    ProcessTable {
        height: auto;
        max-height: 16;
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
        super().__init__(name=name, id=id, classes=classes, disabled=disabled)
        self._columns_added = False

    def on_mount(self) -> None:
        """Add columns when the widget is mounted."""
        if not self._columns_added:
            self.add_columns("PID", "Name", "VRAM", "VRAM%", "Phase", "Rate", "Type")
            self._columns_added = True

    def update_processes(
        self,
        processes: list[GPUProcess],
        phase_states: dict[int, PhaseState],
        total_memory: int,
    ) -> None:
        """Update the table with current process data.

        Args:
            processes: List of GPUProcess objects.
            phase_states: Mapping from PID to PhaseState.
            total_memory: Total GPU memory in bytes.
        """
        self.clear()

        total = max(total_memory, 1)

        # Sort by VRAM descending
        sorted_procs = sorted(processes, key=lambda p: p.used_memory_bytes, reverse=True)

        for proc in sorted_procs:
            pid = proc.identity.pid
            name = _truncate(sanitize_process_name(proc.name))
            vram = _format_bytes(proc.used_memory_bytes)
            vram_pct = f"{proc.used_memory_bytes / total * 100:.1f}%"

            phase_state = phase_states.get(pid)
            if phase_state is not None:
                phase_label = _PHASE_LABELS.get(phase_state.phase, "?")
                rate = f"{phase_state.rate_mb_per_sec:+.1f} MB/s"
            else:
                phase_label = "-"
                rate = "-"

            proc_type = proc.process_type

            self.add_row(str(pid), name, vram, vram_pct, phase_label, rate, proc_type)
