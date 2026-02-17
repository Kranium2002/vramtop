"""Theme management for vramtop.

Provides six built-in themes: dark, light, nord, catppuccin, dracula, solarized.
Themes are TCSS files loaded from this package directory.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_THEMES_DIR = Path(__file__).parent

AVAILABLE_THEMES: dict[str, str] = {
    "dark": "dark.tcss",
    "light": "light.tcss",
    "nord": "nord.tcss",
    "catppuccin": "catppuccin.tcss",
    "dracula": "dracula.tcss",
    "solarized": "solarized.tcss",
}

_DEFAULT_THEME = "dark"


def get_theme_names() -> list[str]:
    """Return sorted list of available theme names."""
    return sorted(AVAILABLE_THEMES.keys())


def load_theme(name: str) -> str:
    """Load theme CSS content by name.

    Falls back to the dark theme if the requested theme is not found
    or cannot be read.

    Args:
        name: Theme name (e.g. 'dark', 'nord').

    Returns:
        CSS content as a string.
    """
    filename = AVAILABLE_THEMES.get(name)
    if filename is None:
        logger.warning("Unknown theme %r, falling back to %r", name, _DEFAULT_THEME)
        filename = AVAILABLE_THEMES[_DEFAULT_THEME]

    theme_path = _THEMES_DIR / filename
    try:
        return theme_path.read_text(encoding="utf-8")
    except OSError:
        logger.warning(
            "Cannot read theme file %s, falling back to %r",
            theme_path,
            _DEFAULT_THEME,
        )
        fallback_path = _THEMES_DIR / AVAILABLE_THEMES[_DEFAULT_THEME]
        return fallback_path.read_text(encoding="utf-8")
