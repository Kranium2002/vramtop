"""Main Textual application for vramtop."""

from __future__ import annotations

import asyncio
import locale
import logging
import os
import time
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.widgets import Footer, Header, Static

from vramtop.analysis.oom_predictor import OOMPredictor
from vramtop.analysis.phase_detector import PhaseDetector, PhaseState
from vramtop.backends.base import BackendError, GPULostError, MemorySnapshot, ProcessIdentity
from vramtop.config import ConfigHolder, VramtopConfig
from vramtop.ui.themes import get_theme_names, load_theme
from vramtop.ui.widgets.detail_panel import DetailPanel
from vramtop.ui.widgets.kill_dialog import KillDialog

if TYPE_CHECKING:
    from vramtop.backends.base import GPUBackend

logger = logging.getLogger(__name__)

# Attempt to import GPU card widget; fall back to a stub if not yet available.
try:
    from vramtop.ui.widgets.gpu_card import GPUCard
except ImportError:  # pragma: no cover
    GPUCard = None  # type: ignore[assignment,misc]

CSS_PATH = Path(__file__).parent / "styles.tcss"

# Enrichment cache TTL in seconds
_ENRICHMENT_TTL = 30.0

# Maximum length for displayed process names before truncation
_MAX_PROCESS_NAME_LEN = 40


class LayoutMode(Enum):
    """Layout modes for the TUI."""

    FULL = "full"
    COMPACT = "compact"
    MINI = "mini"
    TOO_SMALL = "too_small"


def resolve_layout(setting: str, width: int, height: int) -> LayoutMode:
    """Resolve a layout setting to a concrete mode based on terminal size.

    Args:
        setting: Config value — "auto", "full", "compact", or "mini".
        width: Terminal width in columns.
        height: Terminal height in rows.

    Returns:
        The resolved LayoutMode.
    """
    if setting != "auto":
        try:
            return LayoutMode(setting)
        except ValueError:
            pass  # Fall through to auto

    if width >= 120 and height >= 30:
        return LayoutMode.FULL
    if width >= 80 and height >= 20:
        return LayoutMode.COMPACT
    if width >= 40 and height >= 10:
        return LayoutMode.MINI
    return LayoutMode.TOO_SMALL


def format_memory_mb(n: int) -> str:
    """Format bytes as MB with locale-aware thousands separators.

    Example: 4294967296 -> '4,096 MB' (in en_US locale).
    """
    mb = n // (1024 * 1024)
    try:
        formatted = locale.format_string("%d", mb, grouping=True)
    except (ValueError, TypeError):
        formatted = f"{mb:,}"
    return f"{formatted} MB"


def truncate_name(name: str, max_len: int = _MAX_PROCESS_NAME_LEN) -> str:
    """Truncate a process name with '...' if too long."""
    if len(name) <= max_len:
        return name
    return name[: max_len - 3] + "..."


class VramtopApp(App[None]):
    """The htop for GPU memory."""

    CSS_PATH = "styles.tcss"

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("d", "open_detail", "Detail", show=True),
        Binding("k", "open_kill", "Kill", show=True),
        Binding("t", "cycle_theme", "Theme", show=True),
        Binding("s", "save_screenshot", "Screenshot", show=True),
        Binding("question_mark", "show_help", "Help", show=True),
    ]

    def __init__(
        self,
        backend: GPUBackend | None = None,
        config_holder: ConfigHolder | None = None,
        accessible: bool = False,
        csv_path: Path | None = None,
    ) -> None:
        super().__init__()
        self._backend = backend
        self._config_holder = config_holder or ConfigHolder()
        self._csv_path = csv_path
        self._gpu_cards: dict[int, GPUCard | Static] = {}
        # Per-(gpu_index, ProcessIdentity) phase detectors
        self._phase_detectors: dict[tuple[int, ProcessIdentity], PhaseDetector] = {}
        # Latest phase states per (gpu_index, pid)
        self._phase_states: dict[tuple[int, int], PhaseState] = {}
        # Per-GPU OOM predictors and previous used memory
        self._oom_predictors: dict[int, OOMPredictor] = {}
        self._prev_gpu_used_mb: dict[int, float] = {}
        self._last_snapshot: MemorySnapshot | None = None
        self._last_snapshot_time: float = 0.0
        self._poll_timer: object | None = None
        # Enrichment cache: (pid, starttime) -> (enrichment_dict, timestamp)
        self._enrichment_cache: dict[tuple[int, int], tuple[dict[str, object], float]] = {}
        # Layout tracking
        self._layout_mode: LayoutMode = LayoutMode.FULL
        # Theme cycling
        self._theme_names = get_theme_names()
        self._current_theme_idx = 0
        # Set initial theme from config
        theme_name = self._config_holder.config.display.theme
        if theme_name in self._theme_names:
            self._current_theme_idx = self._theme_names.index(theme_name)
        # Accessibility
        self._accessible = accessible
        self._no_color = os.environ.get("NO_COLOR") is not None
        # Loading state — True until first successful poll
        self._first_poll_done = False
        # CSV export
        self._export_manager: object | None = None
        if self._csv_path is not None:
            from vramtop.export import ExportManager

            self._export_manager = ExportManager(
                self._config_holder.config.export,
                csv_path=self._csv_path,
            )
            self._export_manager.start()

    @property
    def config(self) -> VramtopConfig:
        return self._config_holder.config

    @property
    def color_disabled(self) -> bool:
        """Whether color output is suppressed (NO_COLOR env or --accessible)."""
        return self._no_color or self._accessible

    @property
    def accessible(self) -> bool:
        """Whether accessible mode is enabled."""
        return self._accessible

    def compose(self) -> ComposeResult:
        yield Header()
        yield VerticalScroll(
            Static("Loading...", id="loading-msg"),
            id="gpu-container",
        )
        yield DetailPanel(id="detail-panel")
        yield Footer()

    async def on_mount(self) -> None:
        """Initialize backend and start polling."""
        self.title = "vramtop"
        self.sub_title = "GPU Memory Monitor"

        # Apply theme CSS
        self._apply_theme()

        if self._backend is None:
            from vramtop.backends import get_backend

            self._backend = get_backend()
            try:
                await asyncio.to_thread(self._backend.initialize)
            except BackendError as exc:
                logger.error("Backend init failed: %s", exc)
                container = self.query_one("#gpu-container")
                # Remove loading message
                loading = self.query("#loading-msg")
                for w in loading:
                    await w.remove()
                await container.mount(
                    Static(f"[bold red]Error:[/] {exc}", id="error-msg")
                )
                return

        # Take initial snapshot and create cards
        await self._poll()

        # Start periodic polling
        refresh = self.config.general.refresh_rate
        self._poll_timer = self.set_interval(refresh, self._poll)

    def _apply_theme(self) -> None:
        """Load and apply the current theme CSS."""
        if self._no_color:
            return  # Skip theme application when NO_COLOR is set
        theme_name = self._theme_names[self._current_theme_idx]
        css = load_theme(theme_name)
        try:
            self.stylesheet.add_source(css, read_from=(f"theme:{theme_name}", theme_name))
            self.stylesheet.reparse()
            self.refresh_css()
        except Exception:
            logger.debug("Failed to apply theme %r", theme_name, exc_info=True)

    def on_resize(self) -> None:
        """Handle terminal resize — auto-switch layout mode."""
        size = self.size
        setting = self.config.display.layout
        new_mode = resolve_layout(setting, size.width, size.height)
        if new_mode != self._layout_mode:
            self._layout_mode = new_mode
            self._apply_layout()

    def _apply_layout(self) -> None:
        """Adjust widget visibility based on layout mode."""
        if self._layout_mode == LayoutMode.TOO_SMALL:
            container = self.query_one("#gpu-container")
            for card in self._gpu_cards.values():
                card.display = False
            # Show a "too small" message
            try:
                msg = self.query_one("#too-small-msg")
            except Exception:
                msg = None
            if msg is None:
                container.mount(
                    Static(
                        "Terminal too small. Resize to at least 40x10.",
                        id="too-small-msg",
                    )
                )
            return

        # Remove "too small" message if present
        try:
            too_small = self.query_one("#too-small-msg")
            too_small.remove()
        except Exception:
            pass

        for card in self._gpu_cards.values():
            card.display = True

        # Adjust widget visibility per layout mode
        for card in self._gpu_cards.values():
            if GPUCard is not None and isinstance(card, GPUCard):
                try:
                    timeline = card.query_one("Timeline")
                    timeline.display = self._layout_mode == LayoutMode.FULL
                except Exception:
                    pass
                try:
                    proc_table = card.query_one("ProcessTable")
                    if self._layout_mode == LayoutMode.MINI:
                        proc_table.display = False
                    else:
                        proc_table.display = True
                except Exception:
                    pass

    async def _poll(self) -> None:
        """Poll the backend for a new snapshot and update cards."""
        if self._backend is None:
            return

        # Check for config reload (SIGHUP)
        self._config_holder.check_reload()

        try:
            snapshot = await asyncio.to_thread(self._backend.snapshot)
        except GPULostError as exc:
            logger.warning("GPU lost: %s", exc)
            self._mark_lost_gpus()
            return
        except BackendError as exc:
            logger.error("Backend error during poll: %s", exc)
            return

        # Remove loading message on first successful poll
        if not self._first_poll_done:
            self._first_poll_done = True
            try:
                loading = self.query_one("#loading-msg")
                await loading.remove()
            except Exception:
                pass

        now = time.monotonic()
        dt = now - self._last_snapshot_time if self._last_snapshot_time > 0 else 1.0

        # Attempt enrichment for each process
        self._enrich_processes(snapshot, now)

        await self._update_cards(snapshot, dt)

        self._last_snapshot = snapshot
        self._last_snapshot_time = now

        # Push to CSV exporter if active
        if self._export_manager is not None:
            from vramtop.export import ExportManager

            if isinstance(self._export_manager, ExportManager):
                self._export_manager.update_snapshot(snapshot)

    def _enrich_processes(self, snapshot: MemorySnapshot, now: float) -> None:
        """Try to enrich process data; skip gracefully if enrichment module unavailable."""
        try:
            import dataclasses

            from vramtop.enrichment import enrich_process
        except ImportError:
            return

        for device in snapshot.devices:
            for proc in device.processes:
                cache_key = (proc.identity.pid, proc.identity.starttime)
                cached = self._enrichment_cache.get(cache_key)
                if cached is not None and (now - cached[1]) < _ENRICHMENT_TTL:
                    continue
                try:
                    result = enrich_process(
                        proc.identity.pid, proc.identity.starttime
                    )
                    enrichment: dict[str, object] = dataclasses.asdict(result)
                    self._enrichment_cache[cache_key] = (enrichment, now)
                except Exception:
                    logger.debug(
                        "Enrichment failed for PID %d", proc.identity.pid, exc_info=True
                    )

    def _get_enrichment(self, pid: int, starttime: int) -> dict[str, object]:
        """Get cached enrichment data for a process."""
        cached = self._enrichment_cache.get((pid, starttime))
        if cached is not None:
            return cached[0]
        return {}

    async def _update_cards(self, snapshot: MemorySnapshot, dt: float) -> None:
        """Create or update GPU cards from snapshot data."""
        container = self.query_one("#gpu-container")

        for device in snapshot.devices:
            idx = device.index

            # Ensure OOM predictor exists for this GPU
            if idx not in self._oom_predictors:
                oom_cfg = self.config.oom_prediction
                self._oom_predictors[idx] = OOMPredictor(
                    min_sustained_samples=oom_cfg.min_sustained_samples,
                    min_rate_mb_per_sec=oom_cfg.min_rate_mb_per_sec,
                )

            # Update phase detectors for each process
            for proc in device.processes:
                key = (idx, proc.identity)
                if key not in self._phase_detectors:
                    self._phase_detectors[key] = PhaseDetector()

                # Compute per-process delta in MB
                prev_bytes = self._get_prev_process_mem(idx, proc.identity)
                if prev_bytes is not None:
                    delta_mb = (proc.used_memory_bytes - prev_bytes) / (1024 * 1024)
                    state = self._phase_detectors[key].update(delta_mb, dt)
                    self._phase_states[(idx, proc.identity.pid)] = state

            # Gather phase states for this GPU's processes
            phase_states: dict[int, PhaseState] = {
                proc.identity.pid: self._phase_states[(idx, proc.identity.pid)]
                for proc in device.processes
                if (idx, proc.identity.pid) in self._phase_states
            }

            # Get OOM prediction for this GPU (GPU-level, not per-process)
            oom_predictor = self._oom_predictors.get(idx)
            oom_prediction = None
            if oom_predictor is not None:
                used_mb = device.used_memory_bytes / (1024 * 1024)
                free_mb = device.free_memory_bytes / (1024 * 1024)
                total_mb = device.total_memory_bytes / (1024 * 1024)
                prev_used = self._prev_gpu_used_mb.get(idx, used_mb)
                delta_mb = used_mb - prev_used
                self._prev_gpu_used_mb[idx] = used_mb
                oom_prediction = oom_predictor.update(
                    used_mb=used_mb,
                    free_mb=free_mb,
                    total_mb=total_mb,
                    delta_mb=delta_mb,
                    dt_seconds=dt,
                )

            # Create or update the GPU card widget
            if idx not in self._gpu_cards:
                if GPUCard is not None:
                    card: GPUCard | Static = GPUCard(gpu_index=idx, id=f"gpu-{idx}")
                else:
                    card = Static(
                        f"GPU {idx}: {device.name} | "
                        f"{format_memory_mb(device.used_memory_bytes)} / "
                        f"{format_memory_mb(device.total_memory_bytes)} | "
                        f"Util: {device.gpu_util_percent}% | "
                        f"Temp: {device.temperature_celsius}C",
                        id=f"gpu-{idx}",
                    )
                self._gpu_cards[idx] = card
                await container.mount(card)
            else:
                card = self._gpu_cards[idx]

            if GPUCard is not None and isinstance(card, GPUCard):
                card.update_device(device, phase_states, oom_prediction)
            elif isinstance(card, Static):
                card.update(
                    f"GPU {idx}: {device.name} | "
                    f"{format_memory_mb(device.used_memory_bytes)} / "
                    f"{format_memory_mb(device.total_memory_bytes)} | "
                    f"Util: {device.gpu_util_percent}% | "
                    f"Temp: {device.temperature_celsius}C"
                )

    def _get_selected_process(self) -> tuple[int, str, ProcessIdentity] | None:
        """Get the currently selected process from the first GPU card's process table.

        Returns (pid, name, identity) or None.
        """
        if self._last_snapshot is None:
            return None

        # Walk GPUs and find the first highlighted row in a ProcessTable
        for device in self._last_snapshot.devices:
            idx = device.index
            card = self._gpu_cards.get(idx)
            if card is None or not (GPUCard is not None and isinstance(card, GPUCard)):
                continue
            try:
                from vramtop.ui.widgets.process_table import ProcessTable
                proc_table = card.query_one(ProcessTable)
                cursor_row = proc_table.cursor_row
                if cursor_row < 0 or cursor_row >= len(device.processes):
                    continue
                # Processes sorted by VRAM descending in the table
                sorted_procs = sorted(
                    device.processes, key=lambda p: p.used_memory_bytes, reverse=True
                )
                if cursor_row < len(sorted_procs):
                    proc = sorted_procs[cursor_row]
                    return (proc.identity.pid, proc.name, proc.identity)
            except Exception:
                continue

        # Fallback: return first process of first GPU
        for device in self._last_snapshot.devices:
            if device.processes:
                proc = device.processes[0]
                return (proc.identity.pid, proc.name, proc.identity)
        return None

    def action_open_detail(self) -> None:
        """Open the detail panel for the selected process."""
        selected = self._get_selected_process()
        if selected is None:
            return
        pid, name, identity = selected
        enrichment = dict(self._get_enrichment(pid, identity.starttime))

        # Add phase info if available
        for (_gpu_idx, p_id), ps in self._phase_states.items():
            if p_id == pid:
                enrichment.setdefault("phase", ps.phase.value)
                enrichment.setdefault("rate_mb_per_sec", ps.rate_mb_per_sec)
                enrichment.setdefault("confidence", ps.confidence)
                break

        # Add VRAM info from snapshot
        if self._last_snapshot:
            for device in self._last_snapshot.devices:
                for proc in device.processes:
                    if proc.identity == identity:
                        enrichment.setdefault("vram_used_bytes", proc.used_memory_bytes)
                        enrichment.setdefault("vram_total_bytes", device.total_memory_bytes)
                        break

        detail = self.query_one("#detail-panel", DetailPanel)
        from vramtop.sanitize import sanitize_process_name
        detail.show_process(pid, sanitize_process_name(name), enrichment)

    def action_open_kill(self) -> None:
        """Open the kill dialog for the selected process."""
        selected = self._get_selected_process()
        if selected is None:
            return
        pid, name, _identity = selected
        self.open_kill_dialog(pid, name)

    def action_cycle_theme(self) -> None:
        """Cycle to the next available theme."""
        if not self._theme_names:
            return
        self._current_theme_idx = (self._current_theme_idx + 1) % len(self._theme_names)
        self._apply_theme()

    def action_save_screenshot(self) -> None:
        """Save an SVG screenshot to ~/.local/share/vramtop/screenshots/."""
        from vramtop.export.screenshot import take_screenshot

        path = take_screenshot(self)  # type: ignore[arg-type]
        if path is not None:
            self.sub_title = f"Screenshot saved: {path.name}"
        else:
            self.sub_title = "Screenshot failed"

    def action_show_help(self) -> None:
        """Show help info in the subtitle."""
        self.sub_title = "[q]Quit [k]Kill [d]Detail [t]Theme [s]Screenshot [?]Help"

    def open_kill_dialog(self, pid: int, name: str) -> None:
        """Push the kill dialog modal for a given process."""
        enable_kill = self.config.display.enable_kill
        dialog = KillDialog(pid, name, enable_kill=enable_kill)
        self.push_screen(dialog)

    def _get_prev_process_mem(self, gpu_idx: int, identity: ProcessIdentity) -> int | None:
        """Get previous memory usage for a process from the last snapshot."""
        if self._last_snapshot is None:
            return None
        for dev in self._last_snapshot.devices:
            if dev.index == gpu_idx:
                for proc in dev.processes:
                    if proc.identity == identity:
                        return proc.used_memory_bytes
        return None

    def _mark_lost_gpus(self) -> None:
        """Mark all GPU cards as LOST."""
        for idx, card in self._gpu_cards.items():
            if isinstance(card, Static):
                card.update(f"[bold red]GPU {idx}: LOST -- recovery needed[/]")
            elif GPUCard is not None and isinstance(card, GPUCard):
                card.mark_lost()

    async def on_unmount(self) -> None:
        """Clean shutdown: stop backend and exporters."""
        if self._export_manager is not None:
            from vramtop.export import ExportManager

            if isinstance(self._export_manager, ExportManager):
                self._export_manager.stop()
        if self._backend is not None:
            try:
                await asyncio.to_thread(self._backend.shutdown)
            except Exception:
                logger.debug("Backend shutdown error", exc_info=True)
