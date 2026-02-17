"""Space-themed scrollable container with trig-based dot background."""

from __future__ import annotations

import math
from functools import lru_cache
from typing import TYPE_CHECKING

from rich.text import Text
from textual.containers import VerticalScroll

if TYPE_CHECKING:
    from rich.console import RenderableType

# Dot characters ordered by visual "brightness" (faintest to brightest)
_DOT_CHARS = ("\u00b7", "\u2219", "\u22c5", "\u00b0", "\u2727", "\u2726")

# Very dim color so dots sit behind content without distraction
_DOT_COLOR = "#2d333b"


@lru_cache(maxsize=4)
def _generate_pattern(width: int, height: int) -> str:
    """Generate a deterministic dot pattern for the given dimensions.

    Uses overlapping sine/cosine waves to create a sparse, organic-looking
    starfield pattern.  Roughly 3-5% of cells receive a dot character.
    """
    lines: list[str] = []
    for y in range(height):
        chars: list[str] = []
        for x in range(width):
            # Primary density function — two overlapping trig waves.
            # Range is [-2, 2]; threshold 1.65 gives ~3-5% density.
            v = (
                math.sin(x * 0.7) * math.cos(y * 1.3)
                + math.sin(x * 0.3 + y * 0.5)
            )
            if v > 1.65:
                # Secondary function picks which dot character to use
                char_val = (
                    math.sin(x * 1.1 + y * 0.9)
                    * math.cos(x * 0.4 - y * 1.7)
                )
                char_idx = int((char_val + 1.0) * 0.5 * (len(_DOT_CHARS) - 1))
                char_idx = max(0, min(len(_DOT_CHARS) - 1, char_idx))
                chars.append(_DOT_CHARS[char_idx])
            else:
                chars.append(" ")
        lines.append("".join(chars))
    return "\n".join(lines)


class SpaceScroll(VerticalScroll):
    """Scrollable container with a space-themed dot background.

    Dots are visible in the margins between GPU cards, in padding areas,
    and in empty space at the bottom of the scroll region.  Child widgets
    render on top of the pattern naturally.
    """

    def render(self) -> RenderableType:
        """Return the trig-generated dot pattern as background content."""
        width, height = self.size
        if width <= 0 or height <= 0:
            return Text("")
        pattern = _generate_pattern(width, height)
        return Text(pattern, style=_DOT_COLOR)
