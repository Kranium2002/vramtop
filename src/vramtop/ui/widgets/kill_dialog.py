"""Kill confirmation dialog with safety checks and audit logging."""

from __future__ import annotations

import json
import logging
import os
import signal
import time
from pathlib import Path
from typing import TYPE_CHECKING

from textual.binding import Binding
from textual.screen import ModalScreen
from textual.widgets import Button, Static

if TYPE_CHECKING:
    from textual.app import ComposeResult

from vramtop.permissions import is_same_user
from vramtop.process_identity import get_process_identity

# Resolve Docker PID namespace: NVML host PID → container PID.
try:
    from vramtop.enrichment import _resolve_pid
except ImportError:  # pragma: no cover
    def _resolve_pid(pid: int) -> int:  # type: ignore[misc]
        return pid

logger = logging.getLogger(__name__)

_AUDIT_DIR = Path("~/.local/share/vramtop").expanduser()
_AUDIT_LOG = _AUDIT_DIR / "audit.log"

# Delay before offering SIGKILL after SIGTERM
_SIGKILL_WAIT_SECONDS = 5.0


def _ensure_audit_dir() -> None:
    """Create the audit log directory with 0o700 permissions if needed."""
    audit_dir = Path(_AUDIT_DIR)
    if not audit_dir.exists():
        audit_dir.mkdir(parents=True, mode=0o700)
    else:
        # Ensure permissions are correct even if dir already exists
        audit_dir.chmod(0o700)


def _write_audit(entry: dict[str, object]) -> None:
    """Append a JSON-lines entry to the audit log."""
    _ensure_audit_dir()
    entry["timestamp"] = time.time()
    entry["uid"] = os.getuid()
    with open(_AUDIT_LOG, "a") as f:
        f.write(json.dumps(entry, default=str) + "\n")


def _write_audit_entry(
    pid: int, name: str, signal_name: str, success: bool
) -> None:
    """Write a structured audit entry for a kill action."""
    _write_audit({
        "action": f"kill_{signal_name.lower()}",
        "result": "sent" if success else "failed",
        "pid": pid,
        "name": name,
        "signal": signal_name,
    })


def is_kill_allowed(pid: int, enable_kill: bool = True) -> bool:
    """Check if killing a process is allowed.

    Returns True only if enable_kill is True and the process
    belongs to the same user.
    """
    if not enable_kill:
        return False
    return is_same_user(pid)


class KillDialog(ModalScreen[bool]):
    """Modal confirmation for killing a GPU process.

    Safety:
    - Same-UID check before showing
    - (PID, starttime) re-verified before kill
    - SIGTERM first, then offer SIGKILL after 5s
    - All actions audit-logged
    - Respects --no-kill / config.display.enable_kill
    """

    DEFAULT_CSS = """
    KillDialog {
        align: center middle;
    }
    KillDialog #kill-container {
        width: 60;
        height: auto;
        border: thick $error;
        padding: 1 2;
        background: $surface;
    }
    KillDialog #kill-title {
        text-style: bold;
        color: $error;
        margin: 0 0 1 0;
    }
    KillDialog #kill-info {
        margin: 0 0 1 0;
    }
    KillDialog #kill-status {
        margin: 0 0 1 0;
        color: $warning;
    }
    KillDialog .kill-buttons {
        height: 3;
        align: center middle;
    }
    KillDialog Button {
        margin: 0 1;
    }
    """

    BINDINGS = [
        Binding("escape", "cancel", "Cancel", show=True),
    ]

    def __init__(
        self,
        pid: int,
        process_name: str,
        enable_kill: bool = True,
        *,
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
    ) -> None:
        super().__init__(name=name, id=id, classes=classes)
        self._display_pid = pid  # NVML PID (shown in UI)
        # Resolve Docker PID namespace: NVML host PID → container PID
        # for /proc reads, os.kill, and identity checks.
        self._pid = _resolve_pid(pid)
        self._process_name = process_name
        self._enable_kill = enable_kill
        self._sigterm_sent = False
        self._sigterm_time: float = 0.0
        # Capture identity at dialog creation for re-verification
        self._original_identity = get_process_identity(self._pid)

    def compose(self) -> ComposeResult:
        with Static(id="kill-container"):
            yield Static("Kill Process", id="kill-title")
            yield Static("", id="kill-info")
            yield Static("", id="kill-status")
            with Static(classes="kill-buttons"):
                yield Button("Send SIGTERM", id="btn-sigterm", variant="error")
                yield Button("Cancel", id="btn-cancel", variant="default")

    def on_mount(self) -> None:
        """Populate dialog info and run safety checks."""
        info = self.query_one("#kill-info", Static)
        status = self.query_one("#kill-status", Static)
        sigterm_btn = self.query_one("#btn-sigterm", Button)

        # Check --no-kill / config flag
        if not self._enable_kill:
            info.update(
                f"[bold]PID:[/bold] {self._display_pid}\n"
                f"[bold]Name:[/bold] {self._process_name}\n"
            )
            status.update("[bold red]Kill is disabled (--no-kill or config).[/bold red]")
            sigterm_btn.disabled = True
            return

        # Same-UID check (uses resolved container PID)
        if not is_same_user(self._pid):
            info.update(
                f"[bold]PID:[/bold] {self._display_pid}\n"
                f"[bold]Name:[/bold] {self._process_name}\n"
            )
            status.update(
                "[bold red]Cannot kill: process belongs to another user.[/bold red]"
            )
            sigterm_btn.disabled = True
            _write_audit({
                "action": "kill_blocked",
                "reason": "different_uid",
                "pid": self._pid,
                "name": self._process_name,
            })
            return

        # Show process info with age
        identity = self._original_identity
        age_str = "unknown"
        if identity is not None:
            try:
                with open("/proc/uptime") as f:
                    uptime = float(f.read().split()[0])
                clk_tck = os.sysconf("SC_CLK_TCK")
                proc_start_sec = identity.starttime / clk_tck
                age_sec = uptime - proc_start_sec
                if age_sec > 0:
                    if age_sec < 60:
                        age_str = f"{age_sec:.0f}s"
                    elif age_sec < 3600:
                        age_str = f"{age_sec / 60:.0f}m"
                    else:
                        age_str = f"{age_sec / 3600:.1f}h"
            except (OSError, ValueError, IndexError):
                pass

        info.update(
            f"[bold]PID:[/bold] {self._display_pid}\n"
            f"[bold]Name:[/bold] {self._process_name}\n"
            f"[bold]Age:[/bold] {age_str}\n"
            f"[bold]Identity:[/bold] ({identity.pid}, {identity.starttime})"
            if identity else
            f"[bold]PID:[/bold] {self._pid}\n"
            f"[bold]Name:[/bold] {self._process_name}\n"
            f"[bold]Age:[/bold] {age_str}"
        )

        status.update("Press SIGTERM to gracefully stop the process.")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "btn-cancel":
            self.dismiss(False)
        elif event.button.id == "btn-sigterm":
            self._send_sigterm()
        elif event.button.id == "btn-sigkill":
            self._send_sigkill()

    def action_cancel(self) -> None:
        self.dismiss(False)

    def _verify_identity(self) -> bool:
        """Re-verify (PID, starttime) before sending a signal.

        Returns True if the process identity matches, False otherwise.
        """
        if self._original_identity is None:
            return False
        current = get_process_identity(self._pid)
        if current is None:
            return False
        return (
            current.pid == self._original_identity.pid
            and current.starttime == self._original_identity.starttime
        )

    def _send_sigterm(self) -> None:
        """Send SIGTERM after re-verifying process identity."""
        status = self.query_one("#kill-status", Static)

        if not self._verify_identity():
            status.update(
                "[bold red]Process identity changed (PID recycled?). "
                "Aborting kill.[/bold red]"
            )
            _write_audit({
                "action": "kill_aborted",
                "reason": "identity_mismatch",
                "pid": self._pid,
                "name": self._process_name,
            })
            self.query_one("#btn-sigterm", Button).disabled = True
            return

        try:
            os.kill(self._pid, signal.SIGTERM)
        except ProcessLookupError:
            status.update("[bold]Process already exited.[/bold]")
            _write_audit({
                "action": "kill_sigterm",
                "result": "process_gone",
                "pid": self._pid,
                "name": self._process_name,
            })
            self.dismiss(True)
            return
        except PermissionError:
            status.update("[bold red]Permission denied.[/bold red]")
            _write_audit({
                "action": "kill_sigterm",
                "result": "permission_denied",
                "pid": self._pid,
                "name": self._process_name,
            })
            return

        self._sigterm_sent = True
        self._sigterm_time = time.monotonic()

        _write_audit({
            "action": "kill_sigterm",
            "result": "sent",
            "pid": self._pid,
            "name": self._process_name,
            "identity": (
                [self._original_identity.pid, self._original_identity.starttime]
                if self._original_identity else None
            ),
        })

        status.update(
            f"[bold]SIGTERM sent.[/bold] Waiting {_SIGKILL_WAIT_SECONDS:.0f}s... "
            "If the process doesn't exit, SIGKILL will be offered."
        )

        # Replace SIGTERM button with SIGKILL (available after wait)
        sigterm_btn = self.query_one("#btn-sigterm", Button)
        sigterm_btn.disabled = True

        # Schedule offering SIGKILL after the wait period
        self.set_timer(_SIGKILL_WAIT_SECONDS, self._offer_sigkill)

    def _offer_sigkill(self) -> None:
        """After SIGTERM wait, offer SIGKILL if process still exists."""
        identity = get_process_identity(self._pid)
        if identity is None or (
            self._original_identity is not None
            and identity.starttime != self._original_identity.starttime
        ):
            # Process exited or PID recycled
            status = self.query_one("#kill-status", Static)
            status.update("[bold green]Process has exited.[/bold green]")
            self.dismiss(True)
            return

        status = self.query_one("#kill-status", Static)
        status.update(
            "[bold yellow]Process still running after SIGTERM.[/bold yellow]\n"
            "Send SIGKILL (force kill)?"
        )

        # Add SIGKILL button
        buttons = self.query_one(".kill-buttons", Static)
        sigkill_btn = Button("Send SIGKILL", id="btn-sigkill", variant="error")
        buttons.mount(sigkill_btn, before=self.query_one("#btn-cancel"))

    def _send_sigkill(self) -> None:
        """Send SIGKILL after re-verifying identity."""
        status = self.query_one("#kill-status", Static)

        if not self._verify_identity():
            status.update(
                "[bold red]Process identity changed. Aborting.[/bold red]"
            )
            _write_audit({
                "action": "kill_aborted",
                "reason": "identity_mismatch_sigkill",
                "pid": self._pid,
                "name": self._process_name,
            })
            return

        audit_result = "sent"
        try:
            os.kill(self._pid, signal.SIGKILL)
        except ProcessLookupError:
            status.update("[bold]Process already exited.[/bold]")
            audit_result = "process_gone"
        except PermissionError:
            status.update("[bold red]Permission denied for SIGKILL.[/bold red]")
            audit_result = "permission_denied"

        _write_audit({
            "action": "kill_sigkill",
            "result": audit_result,
            "pid": self._pid,
            "name": self._process_name,
            "identity": (
                [self._original_identity.pid, self._original_identity.starttime]
                if self._original_identity else None
            ),
        })

        self.dismiss(True)
