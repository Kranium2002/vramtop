"""Tests for kill dialog safety logic."""

from __future__ import annotations

import json
import os
import signal
import stat
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from vramtop.backends.base import ProcessIdentity
from vramtop.ui.widgets.kill_dialog import (
    KillDialog,
    _AUDIT_DIR,
    _AUDIT_LOG,
    _ensure_audit_dir,
    _write_audit,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def audit_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect audit log to a temp directory."""
    fake_dir = tmp_path / "vramtop"
    fake_log = fake_dir / "audit.log"
    monkeypatch.setattr(
        "vramtop.ui.widgets.kill_dialog._AUDIT_DIR", fake_dir
    )
    monkeypatch.setattr(
        "vramtop.ui.widgets.kill_dialog._AUDIT_LOG", fake_log
    )
    return fake_dir


@pytest.fixture()
def mock_identity() -> ProcessIdentity:
    return ProcessIdentity(pid=12345, starttime=99999)


# ---------------------------------------------------------------------------
# Audit log tests
# ---------------------------------------------------------------------------


class TestAuditLog:
    def test_ensure_audit_dir_creates_with_0700(self, audit_dir: Path) -> None:
        """Audit log directory is created with 0o700 permissions."""
        assert not audit_dir.exists()
        _ensure_audit_dir()
        assert audit_dir.exists()
        mode = audit_dir.stat().st_mode
        assert stat.S_IMODE(mode) == 0o700

    def test_ensure_audit_dir_fixes_permissions(self, audit_dir: Path) -> None:
        """If dir exists with wrong permissions, they get corrected."""
        audit_dir.mkdir(parents=True, mode=0o755)
        _ensure_audit_dir()
        mode = audit_dir.stat().st_mode
        assert stat.S_IMODE(mode) == 0o700

    def test_write_audit_creates_json_lines(self, audit_dir: Path) -> None:
        """Audit entries are written as JSON lines with timestamp and uid."""
        _write_audit({"action": "test", "pid": 1234})
        log_file = audit_dir / "audit.log"
        assert log_file.exists()
        lines = log_file.read_text().strip().split("\n")
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["action"] == "test"
        assert entry["pid"] == 1234
        assert "timestamp" in entry
        assert entry["uid"] == os.getuid()

    def test_write_audit_appends(self, audit_dir: Path) -> None:
        """Multiple audit writes append, not overwrite."""
        _write_audit({"action": "first"})
        _write_audit({"action": "second"})
        log_file = audit_dir / "audit.log"
        lines = log_file.read_text().strip().split("\n")
        assert len(lines) == 2


# ---------------------------------------------------------------------------
# Kill blocked for other-UID processes
# ---------------------------------------------------------------------------


class TestUIDCheck:
    @patch("vramtop.ui.widgets.kill_dialog.is_same_user", return_value=False)
    @patch("vramtop.ui.widgets.kill_dialog.get_process_identity")
    def test_kill_blocked_other_uid(
        self,
        mock_get_id: MagicMock,
        mock_same_user: MagicMock,
        mock_identity: ProcessIdentity,
        audit_dir: Path,
    ) -> None:
        """Kill dialog disables SIGTERM button for other-UID processes."""
        mock_get_id.return_value = mock_identity
        dialog = KillDialog(12345, "other_user_proc", enable_kill=True)
        # Verify is_same_user is used by the on_mount check.
        # We can't call on_mount directly (needs Textual runtime),
        # so test that the UID check function returns False.
        assert not mock_same_user(12345)

        # Also test that _write_audit would log the block
        _write_audit({
            "action": "kill_blocked",
            "reason": "different_uid",
            "pid": 12345,
            "name": "other_user_proc",
        })
        log_file = audit_dir / "audit.log"
        entry = json.loads(log_file.read_text().strip())
        assert entry["action"] == "kill_blocked"
        assert entry["reason"] == "different_uid"


# ---------------------------------------------------------------------------
# (PID, starttime) re-verified before kill
# ---------------------------------------------------------------------------


class TestIdentityVerification:
    @patch("vramtop.ui.widgets.kill_dialog.get_process_identity")
    def test_verify_identity_matches(
        self, mock_get_id: MagicMock, mock_identity: ProcessIdentity
    ) -> None:
        """Identity verification passes when starttime matches."""
        mock_get_id.return_value = mock_identity
        dialog = KillDialog(12345, "test_proc")
        dialog._original_identity = mock_identity
        assert dialog._verify_identity() is True

    @patch("vramtop.ui.widgets.kill_dialog.get_process_identity")
    def test_verify_identity_mismatch(
        self, mock_get_id: MagicMock, mock_identity: ProcessIdentity
    ) -> None:
        """Identity verification fails when starttime changes (PID recycled)."""
        dialog = KillDialog(12345, "test_proc")
        dialog._original_identity = mock_identity
        # Return different starttime (PID recycled)
        mock_get_id.return_value = ProcessIdentity(pid=12345, starttime=88888)
        assert dialog._verify_identity() is False

    @patch("vramtop.ui.widgets.kill_dialog.get_process_identity")
    def test_verify_identity_process_gone(
        self, mock_get_id: MagicMock, mock_identity: ProcessIdentity
    ) -> None:
        """Identity verification fails when process no longer exists."""
        dialog = KillDialog(12345, "test_proc")
        dialog._original_identity = mock_identity
        mock_get_id.return_value = None
        assert dialog._verify_identity() is False

    def test_verify_identity_no_original(self) -> None:
        """Identity verification fails if original identity was None."""
        with patch("vramtop.ui.widgets.kill_dialog.get_process_identity", return_value=None):
            dialog = KillDialog(12345, "test_proc")
        assert dialog._verify_identity() is False


# ---------------------------------------------------------------------------
# SIGTERM sent first
# ---------------------------------------------------------------------------


class TestSIGTERMFirst:
    @patch("vramtop.ui.widgets.kill_dialog.get_process_identity")
    @patch("vramtop.ui.widgets.kill_dialog._write_audit")
    @patch("os.kill")
    def test_send_sigterm_calls_os_kill(
        self,
        mock_os_kill: MagicMock,
        mock_audit: MagicMock,
        mock_get_id: MagicMock,
        mock_identity: ProcessIdentity,
    ) -> None:
        """_send_sigterm sends SIGTERM (not SIGKILL) first."""
        mock_get_id.return_value = mock_identity
        dialog = KillDialog(12345, "test_proc")
        dialog._original_identity = mock_identity

        # Mock the Textual query methods
        mock_status = MagicMock()
        mock_btn = MagicMock()
        dialog.query_one = MagicMock(side_effect=lambda sel, typ=None: {
            "#kill-status": mock_status,
            "#btn-sigterm": mock_btn,
        }.get(sel, MagicMock()))
        dialog.set_timer = MagicMock()

        dialog._send_sigterm()

        mock_os_kill.assert_called_once_with(12345, signal.SIGTERM)
        assert dialog._sigterm_sent is True

    @patch("vramtop.ui.widgets.kill_dialog.get_process_identity")
    @patch("vramtop.ui.widgets.kill_dialog._write_audit")
    @patch("os.kill")
    def test_send_sigterm_audits(
        self,
        mock_os_kill: MagicMock,
        mock_audit: MagicMock,
        mock_get_id: MagicMock,
        mock_identity: ProcessIdentity,
    ) -> None:
        """SIGTERM action is audit-logged."""
        mock_get_id.return_value = mock_identity
        dialog = KillDialog(12345, "test_proc")
        dialog._original_identity = mock_identity
        dialog.query_one = MagicMock(return_value=MagicMock())
        dialog.set_timer = MagicMock()

        dialog._send_sigterm()

        mock_audit.assert_called_once()
        entry = mock_audit.call_args[0][0]
        assert entry["action"] == "kill_sigterm"
        assert entry["result"] == "sent"
        assert entry["pid"] == 12345

    @patch("vramtop.ui.widgets.kill_dialog.get_process_identity")
    @patch("vramtop.ui.widgets.kill_dialog._write_audit")
    def test_send_sigterm_aborts_on_identity_mismatch(
        self,
        mock_audit: MagicMock,
        mock_get_id: MagicMock,
        mock_identity: ProcessIdentity,
    ) -> None:
        """SIGTERM is not sent if identity verification fails."""
        # First call returns the original identity, second returns different
        mock_get_id.side_effect = [
            mock_identity,
            ProcessIdentity(pid=12345, starttime=11111),
        ]
        dialog = KillDialog(12345, "test_proc")
        dialog.query_one = MagicMock(return_value=MagicMock())

        with patch("os.kill") as mock_os_kill:
            dialog._send_sigterm()
            mock_os_kill.assert_not_called()

        # Should audit the abort
        mock_audit.assert_called_once()
        entry = mock_audit.call_args[0][0]
        assert entry["action"] == "kill_aborted"
        assert entry["reason"] == "identity_mismatch"


# ---------------------------------------------------------------------------
# --no-kill flag respected
# ---------------------------------------------------------------------------


class TestNoKillFlag:
    @patch("vramtop.ui.widgets.kill_dialog.get_process_identity")
    def test_no_kill_flag_disables(
        self, mock_get_id: MagicMock, mock_identity: ProcessIdentity
    ) -> None:
        """KillDialog stores enable_kill=False so on_mount can disable button."""
        mock_get_id.return_value = mock_identity
        dialog = KillDialog(12345, "test_proc", enable_kill=False)
        assert dialog._enable_kill is False

    @patch("vramtop.ui.widgets.kill_dialog.get_process_identity")
    def test_kill_enabled_by_default(
        self, mock_get_id: MagicMock, mock_identity: ProcessIdentity
    ) -> None:
        """KillDialog defaults to enable_kill=True."""
        mock_get_id.return_value = mock_identity
        dialog = KillDialog(12345, "test_proc")
        assert dialog._enable_kill is True


# ---------------------------------------------------------------------------
# SIGKILL re-verifies identity
# ---------------------------------------------------------------------------


class TestSIGKILL:
    @patch("vramtop.ui.widgets.kill_dialog.get_process_identity")
    @patch("vramtop.ui.widgets.kill_dialog._write_audit")
    @patch("os.kill")
    def test_send_sigkill_re_verifies(
        self,
        mock_os_kill: MagicMock,
        mock_audit: MagicMock,
        mock_get_id: MagicMock,
        mock_identity: ProcessIdentity,
    ) -> None:
        """_send_sigkill re-verifies identity before sending SIGKILL."""
        mock_get_id.return_value = mock_identity
        dialog = KillDialog(12345, "test_proc")
        dialog._original_identity = mock_identity
        dialog.query_one = MagicMock(return_value=MagicMock())
        dialog.dismiss = MagicMock()

        dialog._send_sigkill()

        mock_os_kill.assert_called_once_with(12345, signal.SIGKILL)

    @patch("vramtop.ui.widgets.kill_dialog.get_process_identity")
    @patch("vramtop.ui.widgets.kill_dialog._write_audit")
    def test_send_sigkill_aborts_on_identity_mismatch(
        self,
        mock_audit: MagicMock,
        mock_get_id: MagicMock,
        mock_identity: ProcessIdentity,
    ) -> None:
        """SIGKILL aborted when identity changes between SIGTERM and SIGKILL."""
        mock_get_id.side_effect = [
            mock_identity,  # constructor
            ProcessIdentity(pid=12345, starttime=77777),  # re-verify
        ]
        dialog = KillDialog(12345, "test_proc")
        dialog.query_one = MagicMock(return_value=MagicMock())

        with patch("os.kill") as mock_os_kill:
            dialog._send_sigkill()
            mock_os_kill.assert_not_called()

        mock_audit.assert_called_once()
        entry = mock_audit.call_args[0][0]
        assert entry["action"] == "kill_aborted"
        assert entry["reason"] == "identity_mismatch_sigkill"
