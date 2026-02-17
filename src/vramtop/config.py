"""Pydantic-validated config with SIGHUP-triggered hot-reload.

Schema matches design doc Section 8 exactly. TOML loading uses
``tomllib`` (3.11+) with ``tomli`` fallback.
"""

from __future__ import annotations

import logging
import signal
import sys
import threading
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator

from vramtop.backends.base import ConfigError

if sys.version_info >= (3, 11):
    import tomllib  # type: ignore[import-not-found]
else:
    try:
        import tomllib  # type: ignore[import-not-found]
    except ModuleNotFoundError:
        import tomli as tomllib

logger = logging.getLogger(__name__)

_CONFIG_DIR = Path("~/.config/vramtop").expanduser()
_DEFAULT_CONFIG_PATH = _CONFIG_DIR / "config.toml"


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class GeneralConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    refresh_rate: float = 1.0
    history_length: int = 300

    @field_validator("refresh_rate")
    @classmethod
    def _check_refresh_rate(cls, v: float) -> float:
        if v < 0.5 or v > 10:
            msg = "refresh_rate must be between 0.5 and 10"
            raise ValueError(msg)
        return v


class AlertsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    oom_warning_seconds: int = 300
    temp_warning_celsius: int = 85
    oom_min_confidence: float = 0.3


class DisplayConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    theme: str = "dark"
    layout: str = "auto"
    show_other_users: bool = False
    enable_kill: bool = True
    phase_indicator: bool = True


class ScrapingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enable: bool = True
    rate_limit_seconds: int = 5
    timeout_ms: int = 500
    allowed_ports: list[int] = [8000, 8080, 11434, 3000]


class OOMPredictionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    algorithm: Literal["variance", "pelt"] = "variance"
    min_sustained_samples: int = 10
    min_rate_mb_per_sec: float = 5.0


class ExportConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prometheus_bind: str = "127.0.0.1"
    prometheus_port: int = 0


class VramtopConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    general: GeneralConfig = GeneralConfig()
    alerts: AlertsConfig = AlertsConfig()
    display: DisplayConfig = DisplayConfig()
    scraping: ScrapingConfig = ScrapingConfig()
    oom_prediction: OOMPredictionConfig = OOMPredictionConfig()
    export: ExportConfig = ExportConfig()


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_config(path: Path | None = None) -> VramtopConfig:
    """Load config from *path*, default locations, or built-in defaults.

    Resolution order:
    1. Explicit *path* (error if missing or invalid).
    2. ``~/.config/vramtop/config.toml`` (skip silently if absent).
    3. Built-in defaults.

    Raises :class:`ConfigError` on parse/validation failure.
    """
    if path is not None:
        return _load_from_path(path)

    if _DEFAULT_CONFIG_PATH.is_file():
        return _load_from_path(_DEFAULT_CONFIG_PATH)

    return VramtopConfig()


def _load_from_path(path: Path) -> VramtopConfig:
    """Parse a TOML file and return a validated config."""
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ConfigError(f"Cannot read config file {path}: {exc}") from exc

    try:
        data = tomllib.loads(raw.decode())
    except (tomllib.TOMLDecodeError, UnicodeDecodeError) as exc:
        raise ConfigError(f"Invalid TOML in {path}: {exc}") from exc

    try:
        return VramtopConfig(**data)
    except Exception as exc:
        raise ConfigError(f"Config validation error in {path}: {exc}") from exc


# ---------------------------------------------------------------------------
# ConfigHolder — runtime config with SIGHUP reload
# ---------------------------------------------------------------------------


class ConfigHolder:
    """Thread-safe config container with signal-triggered reload.

    Usage::

        holder = ConfigHolder(path)
        holder.install_signal_handler()

        # In event loop:
        holder.check_reload()
        cfg = holder.config
    """

    def __init__(self, path: Path | None = None) -> None:
        self._path = path
        self._config = load_config(path)
        self._reload_flag = threading.Event()

    @property
    def config(self) -> VramtopConfig:
        return self._config

    def reload(self) -> None:
        """Reload config from disk.  On failure, keep the old config."""
        try:
            self._config = load_config(self._path)
            logger.info("Config reloaded successfully")
        except ConfigError:
            logger.warning("Config reload failed; keeping previous config", exc_info=True)

    def install_signal_handler(self) -> None:
        """Register SIGHUP to set the reload flag (Unix only)."""
        if not hasattr(signal, "SIGHUP"):
            return
        signal.signal(signal.SIGHUP, self._on_sighup)

    def check_reload(self) -> None:
        """Poll the reload flag; call from the main/event-loop thread."""
        if self._reload_flag.is_set():
            self._reload_flag.clear()
            self.reload()

    # ------------------------------------------------------------------

    def _on_sighup(self, signum: int, frame: object) -> None:
        self._reload_flag.set()
