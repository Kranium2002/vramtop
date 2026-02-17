"""Tests for SVG screenshot export."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from vramtop.export.screenshot import get_screenshots_dir, take_screenshot


class TestGetScreenshotsDir:
    def test_creates_directory(self, tmp_path: Path, monkeypatch: object) -> None:
        import vramtop.export.screenshot as mod

        target = tmp_path / "share" / "vramtop" / "screenshots"
        # Patch expanduser to use tmp_path
        original_expanduser = Path.expanduser

        def fake_expanduser(self: Path) -> Path:
            s = str(self)
            if s.startswith("~"):
                return tmp_path / s[2:]  # strip ~/
            return original_expanduser(self)

        import unittest.mock
        with unittest.mock.patch.object(Path, "expanduser", fake_expanduser):
            result = mod.get_screenshots_dir()

        assert result.exists()
        assert result.is_dir()
        assert "screenshots" in str(result)

    def test_idempotent(self, tmp_path: Path) -> None:
        import unittest.mock
        import vramtop.export.screenshot as mod

        original_expanduser = Path.expanduser

        def fake_expanduser(self: Path) -> Path:
            s = str(self)
            if s.startswith("~"):
                return tmp_path / s[2:]
            return original_expanduser(self)

        with unittest.mock.patch.object(Path, "expanduser", fake_expanduser):
            d1 = mod.get_screenshots_dir()
            d2 = mod.get_screenshots_dir()
        assert d1 == d2


class TestTakeScreenshot:
    def test_saves_svg_file(self, tmp_path: Path) -> None:
        import unittest.mock
        import vramtop.export.screenshot as mod

        app = MagicMock()
        app.export_screenshot.return_value = "<svg>mock</svg>"

        original_expanduser = Path.expanduser

        def fake_expanduser(self: Path) -> Path:
            s = str(self)
            if s.startswith("~"):
                return tmp_path / s[2:]
            return original_expanduser(self)

        with unittest.mock.patch.object(Path, "expanduser", fake_expanduser):
            result = take_screenshot(app)

        assert result is not None
        assert result.exists()
        assert result.suffix == ".svg"
        assert result.read_text() == "<svg>mock</svg>"
        assert "vramtop_" in result.name

    def test_returns_none_on_failure(self) -> None:
        app = MagicMock()
        app.export_screenshot.side_effect = RuntimeError("no terminal")

        result = take_screenshot(app)
        assert result is None

    def test_filename_contains_timestamp(self, tmp_path: Path) -> None:
        import unittest.mock
        import vramtop.export.screenshot as mod

        app = MagicMock()
        app.export_screenshot.return_value = "<svg/>"

        original_expanduser = Path.expanduser

        def fake_expanduser(self: Path) -> Path:
            s = str(self)
            if s.startswith("~"):
                return tmp_path / s[2:]
            return original_expanduser(self)

        with unittest.mock.patch.object(Path, "expanduser", fake_expanduser):
            result = take_screenshot(app)

        assert result is not None
        # Filename format: vramtop_YYYYMMDD_HHMMSS.svg
        name = result.stem  # e.g. "vramtop_20260217_041500"
        assert name.startswith("vramtop_")
        parts = name.split("_")
        assert len(parts) == 3
        assert len(parts[1]) == 8  # YYYYMMDD
        assert len(parts[2]) == 6  # HHMMSS
