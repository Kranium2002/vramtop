"""Unit tests for sanitize module."""

from __future__ import annotations

from vramtop.sanitize import sanitize_process_name


class TestSanitizeProcessName:
    def test_plain_text_unchanged(self) -> None:
        assert sanitize_process_name("python3") == "python3"

    def test_strips_ansi_color(self) -> None:
        assert sanitize_process_name("\x1b[31mred\x1b[0m") == "red"

    def test_strips_ansi_bold(self) -> None:
        assert sanitize_process_name("\x1b[1mbold\x1b[22m") == "bold"

    def test_strips_cursor_movement(self) -> None:
        assert sanitize_process_name("\x1b[2Aup") == "up"

    def test_removes_null_bytes(self) -> None:
        assert sanitize_process_name("py\x00thon") == "python"

    def test_removes_bell(self) -> None:
        assert sanitize_process_name("py\x07thon") == "python"

    def test_removes_backspace(self) -> None:
        assert sanitize_process_name("py\x08thon") == "python"

    def test_removes_del(self) -> None:
        assert sanitize_process_name("py\x7fthon") == "python"

    def test_preserves_tab(self) -> None:
        assert sanitize_process_name("py\tthon") == "py\tthon"

    def test_preserves_newline(self) -> None:
        assert sanitize_process_name("py\nthon") == "py\nthon"

    def test_preserves_carriage_return(self) -> None:
        assert sanitize_process_name("py\rthon") == "py\rthon"

    def test_truncates_long_string(self) -> None:
        result = sanitize_process_name("a" * 500)
        assert len(result) == 256

    def test_empty_string(self) -> None:
        assert sanitize_process_name("") == ""

    def test_only_ansi(self) -> None:
        assert sanitize_process_name("\x1b[31m\x1b[0m") == ""

    def test_idempotent(self) -> None:
        s = "\x1b[31mhello\x00world\x1b[0m" + "x" * 300
        once = sanitize_process_name(s)
        twice = sanitize_process_name(once)
        assert once == twice

    def test_mixed_ansi_and_control(self) -> None:
        s = "\x1b[1m\x07test\x08\x1b[0m"
        assert sanitize_process_name(s) == "test"

    def test_unicode_preserved(self) -> None:
        assert sanitize_process_name("python3 \u2603") == "python3 \u2603"

    def test_osc_sequence_stripped(self) -> None:
        # OSC: ESC ] ... BEL
        assert sanitize_process_name("\x1b]0;title\x07rest") == "rest"
