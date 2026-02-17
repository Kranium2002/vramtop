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
from textual.widgets import Footer, Header, Static

from vramtop.analysis.oom_predictor import OOMPredictor
from vramtop.analysis.phase_detector import PhaseDetector, PhaseState
from vramtop.analysis.survival import (
    SurvivalPrediction,
    check_collective_pressure,
    estimate_peak,
    predict_survival,
)
from vramtop.backends.base import BackendError, GPULostError, MemorySnapshot, ProcessIdentity
from vramtop.config import ConfigHolder, VramtopConfig
from vramtop.ui.themes import get_theme_names, load_theme
from vramtop.ui.widgets.detail_panel import DetailPanel
from vramtop.ui.widgets.kill_dialog import KillDialog
from vramtop.ui.widgets.space_bg import SpaceScroll

if TYPE_CHECKING:
    from vramtop.backends.base import GPUBackend

logger = logging.getLogger(__name__)

# Attempt to import GPU card widget; fall back to a stub if not yet available.
try:
    from vramtop.ui.widgets.gpu_card import GPUCard
except ImportError:  # pragma: no cover
    GPUCard = None  # type: ignore[assignment,misc]

CSS_PATH = Path(__file__).parent / "styles.tcss"

# Enrichment cache TTL in seconds.  Protects against repeated expensive
# /proc reads (maps can be megabytes, fd scan walks hundreds of entries).
# Framework/model/container data is static per process — no need to
# re-detect every second.  Dynamic data (deep mode, scraping) is handled
# separately: detail panel queries deep mode sockets directly, and HTTP
# scrapers have their own rate limiter.
_ENRICHMENT_TTL = 10.0

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
        # Latest phase states per (gpu_index, ProcessIdentity)
        self._phase_states: dict[tuple[int, ProcessIdentity], PhaseState] = {}
        # Per-GPU OOM predictors and previous used memory
        self._oom_predictors: dict[int, OOMPredictor] = {}
        self._prev_gpu_used_mb: dict[int, float] = {}
        self._last_snapshot: MemorySnapshot | None = None
        self._last_snapshot_time: float = 0.0
        self._poll_timer: object | None = None
        # Enrichment cache: (pid, starttime) -> (enrichment_dict, timestamp)
        self._enrichment_cache: dict[tuple[int, int], tuple[dict[str, object], float]] = {}
        # Per-process peak memory tracking: (gpu_index, ProcessIdentity) -> peak bytes
        self._peak_memory: dict[tuple[int, ProcessIdentity], int] = {}
        # Per-process VRAM timeseries for PELT changepoint detection
        # (gpu_index, ProcessIdentity) -> list of used_memory in MB
        self._process_timeseries: dict[tuple[int, ProcessIdentity], list[float]] = {}
        # Saved PELT analysis keyed by sanitized process name.
        # Persists after process exits so the user can review historical analysis.
        # name -> {pelt_phases, pelt_changepoints, pelt_current_phase, ...}
        self._saved_analysis: dict[str, dict[str, object]] = {}
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
        yield SpaceScroll(
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

        # Attempt enrichment for each process (run in thread to avoid
        # blocking the event loop with /proc reads and HTTP scrapes)
        await asyncio.to_thread(self._enrich_processes, snapshot, now)

        await self._update_cards(snapshot, dt)

        # Auto-refresh the detail panel if it's open
        self._refresh_detail_panel()

        # Prune caches for processes no longer in the snapshot
        self._prune_dead_processes(snapshot)

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
                # Don't cache enrichment when starttime=0 (unknown identity).
                # PID recycling could alias different processes under the same
                # (pid, 0) key, causing wrong framework/model associations.
                has_identity = proc.identity.starttime != 0
                cached = self._enrichment_cache.get(cache_key)
                if has_identity and cached is not None and (now - cached[1]) < _ENRICHMENT_TTL:
                    continue
                try:
                    result = enrich_process(
                        proc.identity.pid,
                        proc.identity.starttime,
                        scraping_config=self.config.scraping,
                    )
                    enrichment: dict[str, object] = dataclasses.asdict(result)
                    if has_identity:
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

    def _prune_dead_processes(self, snapshot: MemorySnapshot) -> None:
        """Remove cache entries for processes no longer in the snapshot.

        Prevents unbounded memory growth in long-running sessions where
        many short-lived GPU processes come and go.
        """
        # Build set of all (gpu_index, identity) and (pid, starttime) currently alive
        alive_keys: set[tuple[int, ProcessIdentity]] = set()
        alive_cache_keys: set[tuple[int, int]] = set()
        for device in snapshot.devices:
            for proc in device.processes:
                alive_keys.add((device.index, proc.identity))
                alive_cache_keys.add((proc.identity.pid, proc.identity.starttime))

        # Prune enrichment cache
        for ck in [k for k in self._enrichment_cache if k not in alive_cache_keys]:
            del self._enrichment_cache[ck]

        # Prune phase detectors
        for dk in [k for k in self._phase_detectors if k not in alive_keys]:
            del self._phase_detectors[dk]

        # Prune phase states
        for sk in [k for k in self._phase_states if k not in alive_keys]:
            del self._phase_states[sk]

        # Prune peak memory tracking
        for pk in [k for k in self._peak_memory if k not in alive_keys]:
            del self._peak_memory[pk]

        # Prune VRAM timeseries
        for tk in [k for k in self._process_timeseries if k not in alive_keys]:
            del self._process_timeseries[tk]

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
                    self._phase_states[(idx, proc.identity)] = state

            # Gather phase states for this GPU's processes (keyed by PID for
            # downstream widgets that don't use ProcessIdentity)
            phase_states: dict[int, PhaseState] = {
                proc.identity.pid: self._phase_states[(idx, proc.identity)]
                for proc in device.processes
                if (idx, proc.identity) in self._phase_states
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

            # Compute per-process survival predictions with peak tracking
            survival_states: dict[int, SurvivalPrediction] = {}
            estimated_peaks: dict[int, int] = {}
            for proc in device.processes:
                pid = proc.identity.pid
                peak_key = (idx, proc.identity)

                # Track historical peak memory per process
                prev_peak = self._peak_memory.get(peak_key, 0)
                current_peak = max(prev_peak, proc.used_memory_bytes)
                self._peak_memory[peak_key] = current_peak

                # Accumulate VRAM timeseries for PELT analysis
                ts = self._process_timeseries.setdefault(peak_key, [])
                ts.append(proc.used_memory_bytes / (1024 * 1024))
                # Cap at 1000 samples (~16 min at 1s refresh) to bound memory
                if len(ts) > 1000:
                    del ts[: len(ts) - 1000]

                enrichment = self._get_enrichment(
                    proc.identity.pid, proc.identity.starttime
                )
                phase_state = phase_states.get(pid)
                phase_str = phase_state.phase.value if phase_state else "stable"
                framework = enrichment.get("framework")
                cmdline = enrichment.get("cmdline")
                model_size_raw = enrichment.get("estimated_model_size_bytes")
                model_size_bytes: int | None = (
                    int(str(model_size_raw)) if model_size_raw is not None else None
                )
                scrape_data = enrichment.get("scrape_data")
                fw_str = str(framework) if framework else None
                cmd_str = str(cmdline) if cmdline else None

                survival_states[pid] = predict_survival(
                    phase=phase_str,
                    framework=fw_str,
                    cmdline=cmd_str,
                    model_size_bytes=model_size_bytes,
                    process_used_bytes=proc.used_memory_bytes,
                    gpu_free_bytes=device.free_memory_bytes,
                    gpu_total_bytes=device.total_memory_bytes,
                    scrape_data=dict(scrape_data) if isinstance(scrape_data, dict) else None,
                    peak_used_bytes=current_peak if current_peak > 0 else None,
                )

                # Track estimated peak for collective pressure check
                estimated_peaks[pid] = estimate_peak(
                    framework=fw_str,
                    cmdline=cmd_str,
                    model_size_bytes=model_size_bytes,
                    process_used_bytes=proc.used_memory_bytes,
                    peak_used_bytes=current_peak if current_peak > 0 else None,
                )

            # Collective pressure: if sum of estimated peaks exceeds GPU total,
            # upgrade individual verdicts
            survival_states = check_collective_pressure(
                survival_states, estimated_peaks, device.total_memory_bytes
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
                # Build per-PID enrichment dict for the process table
                proc_enrichments: dict[int, dict[str, object]] = {}
                for proc in device.processes:
                    enr = self._get_enrichment(
                        proc.identity.pid, proc.identity.starttime
                    )
                    if enr:
                        proc_enrichments[proc.identity.pid] = enr

                card.update_device(
                    device, phase_states, oom_prediction,
                    survival_states=survival_states,
                    enrichments=proc_enrichments if proc_enrichments else None,
                )
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

    def _build_detail_enrichment(
        self, pid: int, name: str, identity: ProcessIdentity
    ) -> dict[str, object]:
        """Build the enrichment dict for the detail panel."""
        enrichment = dict(self._get_enrichment(pid, identity.starttime))

        # Add phase info if available
        for (_gpu_idx, p_identity), ps in self._phase_states.items():
            if p_identity == identity:
                enrichment["phase"] = ps.phase.value
                enrichment["rate_mb_per_sec"] = ps.rate_mb_per_sec
                enrichment["confidence"] = ps.confidence
                break

        # Add VRAM info from snapshot
        if self._last_snapshot:
            for device in self._last_snapshot.devices:
                for proc in device.processes:
                    if proc.identity == identity:
                        enrichment["vram_used_bytes"] = proc.used_memory_bytes
                        enrichment["vram_total_bytes"] = device.total_memory_bytes
                        break

        # Always query deep mode live (bypass enrichment cache).
        # The reporter socket updates every second — much faster than the
        # enrichment cache TTL.  Merge into existing scrape_data.
        try:
            from vramtop.enrichment.deep_mode import get_deep_enrichment

            deep = get_deep_enrichment(pid)
            if deep is not None:
                existing = enrichment.get("scrape_data")
                if isinstance(existing, dict):
                    existing.update(deep)
                    enrichment["scrape_data"] = existing
                else:
                    enrichment["scrape_data"] = deep
        except Exception:
            pass

        # PELT changepoint analysis on accumulated timeseries.
        # Always runs in the detail panel when enough samples are
        # collected and ruptures is installed.  Falls back gracefully.
        try:
            self._add_pelt_analysis(
                enrichment, identity, process_name=name,
            )
        except Exception:
            logger.debug("PELT analysis failed", exc_info=True)

        # Save full enrichment by process name so it survives process exit.
        # The detail panel can then show post-mortem data for exited processes.
        if name:
            self._saved_analysis[name] = dict(enrichment)

        return enrichment

    def _add_pelt_analysis(
        self, enrichment: dict[str, object], identity: ProcessIdentity,
        process_name: str = "",
    ) -> None:
        """Run PELT changepoint detection and add results to enrichment."""
        from vramtop.analysis.pelt_detector import PELTDetector, get_preset_penalty
        from vramtop.analysis.segment_labels import compute_segment_stats

        # Find the timeseries for this process (any GPU index)
        ts: list[float] | None = None
        for (_, p_identity), series in self._process_timeseries.items():
            if p_identity == identity:
                ts = series
                break

        if ts is None or len(ts) < 10:
            return

        fw = enrichment.get("framework")
        fw_str = str(fw) if fw else None
        penalty = get_preset_penalty(fw_str)
        detector = PELTDetector(penalty=penalty, min_size=10)

        changepoints = detector.detect_changepoints(ts)
        phases = detector.classify_segments(ts, changepoints)

        if not phases:
            return

        enrichment["pelt_changepoints"] = changepoints
        enrichment["pelt_phases"] = [p.value for p in phases]
        enrichment["pelt_current_phase"] = phases[-1].value
        enrichment["pelt_num_samples"] = len(ts)

        # Compute labeled segment statistics for chart + summary
        vram_total_raw = enrichment.get("vram_total_bytes")
        gpu_total_mb = (
            float(vram_total_raw) / (1024 * 1024)
            if isinstance(vram_total_raw, (int, float))
            else 0.0
        )
        segments = compute_segment_stats(ts, changepoints, phases, gpu_total_mb=gpu_total_mb)
        if segments:
            enrichment["pelt_segments"] = segments
            enrichment["pelt_timeseries"] = list(ts)

    def _refresh_detail_panel(self) -> None:
        """Auto-refresh the detail panel if it's visible.

        If the process is still alive, refreshes with latest data.
        If the process has exited, shows last known data with [EXITED] tag
        so deep mode / enrichment data is still visible.
        """
        detail = self.query_one("#detail-panel", DetailPanel)
        if not detail.has_class("visible") or detail.current_pid is None:
            return

        target_pid = detail.current_pid

        # Try to find the process in the current snapshot
        if self._last_snapshot is not None:
            for device in self._last_snapshot.devices:
                for proc in device.processes:
                    if proc.identity.pid == target_pid:
                        from vramtop.sanitize import sanitize_process_name

                        enrichment = self._build_detail_enrichment(
                            proc.identity.pid, proc.name, proc.identity
                        )
                        detail.show_process(
                            proc.identity.pid,
                            sanitize_process_name(proc.name),
                            enrichment,
                        )
                        return

        # Process has exited — show last known data with [EXITED] label.
        # Don't clear the panel; the user explicitly opened it and the
        # enrichment/deep mode data is still valuable for post-mortem.
        detail.mark_exited()

    def action_open_detail(self) -> None:
        """Open the detail panel for the selected process."""
        selected = self._get_selected_process()
        if selected is None:
            return
        pid, name, identity = selected
        enrichment = self._build_detail_enrichment(pid, name, identity)
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
