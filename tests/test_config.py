"""Unit tests for config module."""

from __future__ import annotations

import os
import signal
import textwrap
from pathlib import Path

import pytest
from pydantic import ValidationError

from vramtop.backends.base import ConfigError
from vramtop.config import (
    ConfigHolder,
    GeneralConfig,
    VramtopConfig,
    load_config,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_toml(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "config.toml"
    p.write_text(textwrap.dedent(content))
    return p


# ---------------------------------------------------------------------------
# load_config — valid TOML
# ---------------------------------------------------------------------------


class TestLoadConfigValid:
    def test_full_config(self, tmp_path: Path) -> None:
        path = _write_toml(
            tmp_path,
            """\
            [general]
            refresh_rate = 2.0
            history_length = 500

            [alerts]
            oom_warning_seconds = 120
            temp_warning_celsius = 90
            oom_min_confidence = 0.5

            [display]
            theme = "nord"
            layout = "compact"
            show_other_users = true
            enable_kill = false
            phase_indicator = false

            [scraping]
            enable = false
            rate_limit_seconds = 10
            timeout_ms = 1000
            allowed_ports = [8000]

            [oom_prediction]
            algorithm = "pelt"
            min_sustained_samples = 20
            min_rate_mb_per_sec = 2.5

            [export]
            prometheus_bind = "0.0.0.0"
            prometheus_port = 9090
            """,
        )
        cfg = load_config(path)
        assert cfg.general.refresh_rate == 2.0
        assert cfg.general.history_length == 500
        assert cfg.alerts.oom_warning_seconds == 120
        assert cfg.display.theme == "nord"
        assert cfg.display.layout == "compact"
        assert cfg.display.show_other_users is True
        assert cfg.display.enable_kill is False
        assert cfg.scraping.enable is False
        assert cfg.scraping.allowed_ports == [8000]
        assert cfg.oom_prediction.algorithm == "pelt"
        assert cfg.oom_prediction.min_sustained_samples == 20
        assert cfg.export.prometheus_port == 9090

    def test_partial_config_uses_defaults(self, tmp_path: Path) -> None:
        """Only some sections present; rest should get defaults."""
        path = _write_toml(
            tmp_path,
            """\
            [general]
            refresh_rate = 3.0
            """,
        )
        cfg = load_config(path)
        assert cfg.general.refresh_rate == 3.0
        assert cfg.general.history_length == 300  # default
        assert cfg.alerts.oom_warning_seconds == 300  # default
        assert cfg.display.theme == "dark"  # default
        assert cfg.scraping.enable is True  # default
        assert cfg.oom_prediction.algorithm == "variance"  # default
        assert cfg.export.prometheus_port == 0  # default

    def test_empty_file_gives_defaults(self, tmp_path: Path) -> None:
        path = _write_toml(tmp_path, "")
        cfg = load_config(path)
        assert cfg == VramtopConfig()


# ---------------------------------------------------------------------------
# load_config — missing file -> defaults
# ---------------------------------------------------------------------------


class TestLoadConfigMissingFile:
    def test_none_path_no_default_file(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """No explicit path and no default config file -> built-in defaults."""
        monkeypatch.setattr(
            "vramtop.config._DEFAULT_CONFIG_PATH",
            Path("/nonexistent/path/config.toml"),
        )
        cfg = load_config(None)
        assert cfg == VramtopConfig()


# ---------------------------------------------------------------------------
# load_config — unknown keys rejected
# ---------------------------------------------------------------------------


class TestLoadConfigUnknownKeys:
    def test_unknown_top_level_key(self, tmp_path: Path) -> None:
        path = _write_toml(
            tmp_path,
            """\
            [bogus]
            key = "value"
            """,
        )
        with pytest.raises(ConfigError, match="validation error"):
            load_config(path)

    def test_unknown_nested_key(self, tmp_path: Path) -> None:
        path = _write_toml(
            tmp_path,
            """\
            [general]
            refresh_rate = 1.0
            unknown_field = 42
            """,
        )
        with pytest.raises(ConfigError, match="validation error"):
            load_config(path)


# ---------------------------------------------------------------------------
# load_config — out-of-range refresh_rate
# ---------------------------------------------------------------------------


class TestLoadConfigRefreshRate:
    def test_too_low(self, tmp_path: Path) -> None:
        path = _write_toml(
            tmp_path,
            """\
            [general]
            refresh_rate = 0.1
            """,
        )
        with pytest.raises(ConfigError, match="validation error"):
            load_config(path)

    def test_too_high(self, tmp_path: Path) -> None:
        path = _write_toml(
            tmp_path,
            """\
            [general]
            refresh_rate = 20.0
            """,
        )
        with pytest.raises(ConfigError, match="validation error"):
            load_config(path)

    def test_boundary_low(self, tmp_path: Path) -> None:
        path = _write_toml(
            tmp_path,
            """\
            [general]
            refresh_rate = 0.5
            """,
        )
        cfg = load_config(path)
        assert cfg.general.refresh_rate == 0.5

    def test_boundary_high(self, tmp_path: Path) -> None:
        path = _write_toml(
            tmp_path,
            """\
            [general]
            refresh_rate = 10.0
            """,
        )
        cfg = load_config(path)
        assert cfg.general.refresh_rate == 10.0


# ---------------------------------------------------------------------------
# load_config — corrupt TOML
# ---------------------------------------------------------------------------


class TestLoadConfigCorrupt:
    def test_invalid_toml_syntax(self, tmp_path: Path) -> None:
        path = _write_toml(tmp_path, "[general\nrefresh_rate = ???")
        with pytest.raises(ConfigError, match="Invalid TOML"):
            load_config(path)

    def test_explicit_path_missing_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "does_not_exist.toml"
        with pytest.raises(ConfigError, match="Cannot read"):
            load_config(path)


# ---------------------------------------------------------------------------
# Pydantic model direct validation
# ---------------------------------------------------------------------------


class TestModelValidation:
    def test_general_extra_forbid(self) -> None:
        with pytest.raises(ValidationError):
            GeneralConfig(unknown=1)  # type: ignore[call-arg]

    def test_refresh_rate_validator_low(self) -> None:
        with pytest.raises(ValidationError, match="refresh_rate"):
            GeneralConfig(refresh_rate=0.1)

    def test_refresh_rate_validator_high(self) -> None:
        with pytest.raises(ValidationError, match="refresh_rate"):
            GeneralConfig(refresh_rate=11.0)


# ---------------------------------------------------------------------------
# ConfigHolder — SIGHUP reload
# ---------------------------------------------------------------------------


class TestConfigHolderSIGHUP:
    @pytest.mark.skipif(
        not hasattr(signal, "SIGHUP"),
        reason="SIGHUP not available on this platform",
    )
    def test_sighup_sets_flag(self, tmp_path: Path) -> None:
        path = _write_toml(
            tmp_path,
            """\
            [general]
            refresh_rate = 1.0
            """,
        )
        holder = ConfigHolder(path)
        holder.install_signal_handler()

        assert not holder._reload_flag.is_set()
        os.kill(os.getpid(), signal.SIGHUP)
        assert holder._reload_flag.is_set()

    @pytest.mark.skipif(
        not hasattr(signal, "SIGHUP"),
        reason="SIGHUP not available on this platform",
    )
    def test_check_reload_clears_flag_and_reloads(self, tmp_path: Path) -> None:
        path = _write_toml(
            tmp_path,
            """\
            [general]
            refresh_rate = 1.0
            """,
        )
        holder = ConfigHolder(path)
        holder.install_signal_handler()

        assert holder.config.general.refresh_rate == 1.0

        # Update file and trigger SIGHUP
        path.write_text(
            textwrap.dedent(
                """\
                [general]
                refresh_rate = 5.0
                """
            )
        )
        os.kill(os.getpid(), signal.SIGHUP)
        holder.check_reload()

        assert holder.config.general.refresh_rate == 5.0
        assert not holder._reload_flag.is_set()

    def test_reload_keeps_old_on_bad_file(self, tmp_path: Path) -> None:
        path = _write_toml(
            tmp_path,
            """\
            [general]
            refresh_rate = 2.0
            """,
        )
        holder = ConfigHolder(path)
        assert holder.config.general.refresh_rate == 2.0

        # Corrupt the file
        path.write_text("[general\nbad!!!")
        holder.reload()

        # Old config preserved
        assert holder.config.general.refresh_rate == 2.0

    def test_holder_defaults_when_no_path(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "vramtop.config._DEFAULT_CONFIG_PATH",
            Path("/nonexistent/path/config.toml"),
        )
        holder = ConfigHolder()
        assert holder.config == VramtopConfig()
