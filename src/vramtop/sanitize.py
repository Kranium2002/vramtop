"""Input sanitization for process names and external strings."""

from __future__ import annotations

import re

_MAX_LENGTH = 256

# ANSI escape sequences: ESC[ ... final byte, or ESC followed by other sequences
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]|\x1b[()][AB012]|\x1b\].*?\x07|\x1b[^[\]()]")

# Control characters: anything < 0x20 except tab (0x09), newline (0x0a), carriage return (0x0d),
# plus DEL (0x7f)
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def sanitize_process_name(name: str) -> str:
    """Strip ANSI escapes, control chars, and truncate.

    Idempotent: sanitize(sanitize(x)) == sanitize(x) for all x.
    """
    result = _ANSI_RE.sub("", name)
    result = _CONTROL_RE.sub("", result)
    return result[:_MAX_LENGTH]
