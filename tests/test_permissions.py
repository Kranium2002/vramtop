"""Unit tests for permissions module (mocked /proc)."""

from __future__ import annotations

import os
from unittest.mock import patch

from vramtop.permissions import check_proc_readable, is_same_user


class TestIsSameUser:
    def test_same_user(self) -> None:
        """Current process should be same user."""
        assert is_same_user(os.getpid()) is True

    def test_nonexistent_pid(self) -> None:
        assert is_same_user(999999999) is False

    def test_permission_error(self) -> None:
        with patch("vramtop.permissions.os.stat", side_effect=PermissionError):
            assert is_same_user(1) is False

    def test_different_uid(self) -> None:
        class FakeStat:
            st_uid = 99999

        with patch("vramtop.permissions.os.stat", return_value=FakeStat()):
            assert is_same_user(1) is False

    def test_file_not_found(self) -> None:
        with patch("vramtop.permissions.os.stat", side_effect=FileNotFoundError):
            assert is_same_user(1) is False


class TestCheckProcReadable:
    def test_current_process_readable(self) -> None:
        assert check_proc_readable(os.getpid()) is True

    def test_nonexistent_pid(self) -> None:
        assert check_proc_readable(999999999) is False

    def test_permission_error(self) -> None:
        with patch("vramtop.permissions.os.listdir", side_effect=PermissionError):
            assert check_proc_readable(1) is False

    def test_file_not_found(self) -> None:
        with patch("vramtop.permissions.os.listdir", side_effect=FileNotFoundError):
            assert check_proc_readable(1) is False
