"""Process identity using (PID, starttime) to defend against PID recycling."""

from __future__ import annotations

from vramtop.backends.base import ProcessIdentity


def get_process_identity(pid: int) -> ProcessIdentity | None:
    """Return ProcessIdentity or None if process doesn't exist.

    Parses /proc/<pid>/stat field 22 (starttime). Uses rfind(')')
    to handle comm fields containing spaces or parentheses.
    """
    try:
        with open(f"/proc/{pid}/stat") as f:
            stat = f.read()
        end_comm = stat.rfind(")")
        fields = stat[end_comm + 2 :].split()
        starttime = int(fields[19])  # field 22, index 19 after comm
        return ProcessIdentity(pid=pid, starttime=starttime)
    except (FileNotFoundError, ProcessLookupError, IndexError, ValueError):
        return None
