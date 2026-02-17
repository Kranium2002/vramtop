"""Hypothesis property-based tests for sanitize module."""

from __future__ import annotations

from hypothesis import given, strategies as st

from vramtop.sanitize import sanitize_process_name


@given(text=st.text(min_size=0, max_size=500))
def test_idempotent(text: str) -> None:
    """sanitize(sanitize(x)) == sanitize(x) for all text."""
    once = sanitize_process_name(text)
    twice = sanitize_process_name(once)
    assert once == twice


@given(text=st.text(min_size=0, max_size=500))
def test_max_length(text: str) -> None:
    """Result is always <= 256 chars."""
    assert len(sanitize_process_name(text)) <= 256


@given(text=st.text(min_size=0, max_size=500))
def test_no_control_chars(text: str) -> None:
    """No control characters in result (except tab, newline, CR)."""
    result = sanitize_process_name(text)
    for ch in result:
        code = ord(ch)
        if code < 0x20:
            assert code in (0x09, 0x0A, 0x0D), f"Control char U+{code:04X} found in result"
        assert code != 0x7F, "DEL char found in result"


@given(text=st.text(min_size=0, max_size=500))
def test_no_ansi_escape(text: str) -> None:
    """No ESC character in result."""
    result = sanitize_process_name(text)
    assert "\x1b" not in result
