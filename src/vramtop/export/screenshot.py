"""SVG screenshot export using Textual's built-in screenshot support."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from textual.app import App

logger = logging.getLogger(__name__)

_SCREENSHOTS_DIR_NAME = "screenshots"


def get_screenshots_dir() -> Path:
    """Return ``~/.local/share/vramtop/screenshots/``, creating it if needed."""
    base = Path("~/.local/share/vramtop").expanduser()
    d = base / _SCREENSHOTS_DIR_NAME
    d.mkdir(parents=True, exist_ok=True)
    return d


def take_screenshot(app: App[object]) -> Path | None:
    """Save an SVG screenshot and return the file path, or *None* on failure."""
    try:
        d = get_screenshots_dir()
        ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"vramtop_{ts}.svg"
        filepath = d / filename

        svg = app.export_screenshot()
        filepath.write_text(svg, encoding="utf-8")
        logger.info("Screenshot saved to %s", filepath)
        return filepath
    except Exception:
        logger.warning("Failed to save screenshot", exc_info=True)
        return None
