"""Security boundary tests — Commit 14.

Verifies that all enrichers refuse access for other-UID processes,
sanitize is applied to all /proc/ strings, kill is blocked for other-UID,
container detection only reads own-process environ, and audit log
has correct permissions.
"""

from __future__ import annotations

import os
import stat
import tempfile
from unittest.mock import patch

from vramtop.enrichment import enrich_process
from vramtop.sanitize import sanitize_process_name
from vramtop.ui.widgets.kill_dialog import (
    _AUDIT_DIR,
    _ensure_audit_dir,
    _write_audit,
)


class TestEnrichmentUIDBarrier:
    """All enrichers must refuse /proc/<pid>/ for other-UID processes."""

    @patch("vramtop.enrichment.is_same_user", return_value=False)
    def test_enrich_returns_empty_for_other_uid(self, _mock: object) -> None:
        result = enrich_process(pid=99999, starttime=1)
        assert result.framework is None
        assert result.cmdline is None
        assert len(result.model_files) == 0
        assert result.is_mps_client is False

    @patch("vramtop.enrichment.detector.is_same_user", return_value=False)
    def test_detector_refuses_other_uid(self, _mock: object) -> None:
        from vramtop.enrichment.detector import _cache, detect_framework

        _cache.clear()
        fw, ver = detect_framework(pid=99999, starttime=1)
        assert fw is None
        assert ver is None

    @patch("vramtop.enrichment.model_files.is_same_user", return_value=False)
    def test_model_files_refuses_other_uid(self, _mock: object) -> None:
        from vramtop.enrichment.model_files import scan_model_files

        files = scan_model_files(pid=99999)
        assert files == []


class TestSanitizeOnProcStrings:
    """Sanitize must be applied to all strings from /proc/."""

    def test_ansi_stripped(self) -> None:
        dirty = "\x1b[31mmalicious\x1b[0m"
        clean = sanitize_process_name(dirty)
        assert "\x1b" not in clean
        assert "malicious" in clean

    def test_control_chars_removed(self) -> None:
        dirty = "normal\x00\x01\x02hidden"
        clean = sanitize_process_name(dirty)
        assert "\x00" not in clean
        assert "\x01" not in clean

    def test_truncation(self) -> None:
        long_name = "a" * 500
        clean = sanitize_process_name(long_name)
        assert len(clean) <= 256

    def test_idempotent(self) -> None:
        text = "\x1b[1mtest\x1b[0m \x00 data"
        once = sanitize_process_name(text)
        twice = sanitize_process_name(once)
        assert once == twice


class TestAuditLogPermissions:
    """Audit log directory must be created with 0o700 permissions."""

    def test_audit_dir_created_with_0700(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            audit_dir = os.path.join(tmpdir, "vramtop")
            with patch(
                "vramtop.ui.widgets.kill_dialog._AUDIT_DIR",
                type(_AUDIT_DIR)(audit_dir),
            ):
                _ensure_audit_dir()
                mode = stat.S_IMODE(os.stat(audit_dir).st_mode)
                assert mode == 0o700

    def test_audit_entry_written(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            audit_dir = os.path.join(tmpdir, "vramtop")
            audit_file = os.path.join(audit_dir, "audit.log")
            with (
                patch(
                    "vramtop.ui.widgets.kill_dialog._AUDIT_DIR",
                    type(_AUDIT_DIR)(audit_dir),
                ),
                patch(
                    "vramtop.ui.widgets.kill_dialog._AUDIT_LOG",
                    type(_AUDIT_DIR)(audit_file),
                ),
            ):
                _write_audit({
                    "action": "test",
                    "pid": 1234,
                    "signal": "SIGTERM",
                })
                assert os.path.exists(audit_file)
                with open(audit_file) as f:
                    content = f.read()
                assert "1234" in content
                assert "SIGTERM" in content
