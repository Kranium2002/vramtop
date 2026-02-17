"""Unit tests for process_identity module (mocked /proc/pid/stat)."""

from __future__ import annotations

import os
from unittest.mock import mock_open, patch

from vramtop.backends.base import ProcessIdentity
from vramtop.process_identity import get_process_identity


# Example /proc/<pid>/stat content:
# "1234 (python3) S 1233 1234 1234 0 -1 ... <field22=starttime> ..."
# Fields after ')': state, ppid, pgrp, session, tty_nr, tpgid, flags,
#   minflt, cminflt, majflt, cmajflt, utime, stime, cutime, cstime,
#   priority, nice, num_threads, itrealvalue, starttime(index19), ...

def _make_stat(pid: int, comm: str, starttime: int) -> str:
    """Build a minimal /proc/<pid>/stat line."""
    # 19 fields before starttime (after comm): indices 0-18, then starttime at 19
    fields_before = " ".join(["S", "1", str(pid), str(pid), "0", "-1", "0"] + ["0"] * 12)
    return f"{pid} ({comm}) {fields_before} {starttime} 0 0"


class TestGetProcessIdentity:
    def test_normal_process(self) -> None:
        stat = _make_stat(1234, "python3", 98765)
        with patch("builtins.open", mock_open(read_data=stat)):
            result = get_process_identity(1234)
        assert result == ProcessIdentity(pid=1234, starttime=98765)

    def test_comm_with_spaces(self) -> None:
        stat = _make_stat(5678, "my process", 11111)
        with patch("builtins.open", mock_open(read_data=stat)):
            result = get_process_identity(5678)
        assert result == ProcessIdentity(pid=5678, starttime=11111)

    def test_comm_with_parens(self) -> None:
        stat = _make_stat(9999, "foo (bar)", 22222)
        with patch("builtins.open", mock_open(read_data=stat)):
            result = get_process_identity(9999)
        assert result == ProcessIdentity(pid=9999, starttime=22222)

    def test_process_not_found(self) -> None:
        with patch("builtins.open", side_effect=FileNotFoundError):
            assert get_process_identity(999999999) is None

    def test_process_lookup_error(self) -> None:
        with patch("builtins.open", side_effect=ProcessLookupError):
            assert get_process_identity(123) is None

    def test_malformed_stat(self) -> None:
        with patch("builtins.open", mock_open(read_data="garbage")):
            assert get_process_identity(1) is None

    def test_too_few_fields(self) -> None:
        with patch("builtins.open", mock_open(read_data="1 (x) S 1 2")):
            assert get_process_identity(1) is None

    def test_current_process(self) -> None:
        """Smoke test against real /proc for current process."""
        result = get_process_identity(os.getpid())
        assert result is not None
        assert result.pid == os.getpid()
        assert result.starttime > 0
