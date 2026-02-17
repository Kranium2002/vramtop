"""CSV export — append one row per (GPU, process) per snapshot."""

from __future__ import annotations

import csv
import logging
import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import io
    from pathlib import Path

    from vramtop.backends.base import MemorySnapshot

logger = logging.getLogger(__name__)

_CSV_COLUMNS = [
    "wall_time",
    "gpu_index",
    "gpu_name",
    "gpu_used_bytes",
    "gpu_total_bytes",
    "gpu_util_percent",
    "pid",
    "process_name",
    "process_vram_bytes",
    "process_type",
]


class CSVLogger:
    """Appends GPU/process rows to a CSV file on each snapshot update.

    Thread-safe: writes are guarded by a lock.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._file: io.TextIOWrapper | None = None
        self._writer: csv.DictWriter[str] | None = None

    # -- lifecycle --

    def start(self) -> None:
        """Open the CSV file and write the header row."""
        with self._lock:
            if self._file is not None:
                return
            self._file = self._path.open("a", newline="", encoding="utf-8")
            self._writer = csv.DictWriter(self._file, fieldnames=_CSV_COLUMNS)
            # Write header only if file is empty / newly created.
            if self._path.stat().st_size == 0 or self._file.tell() == 0:
                self._writer.writeheader()
                self._file.flush()

    def stop(self) -> None:
        """Flush and close the file."""
        with self._lock:
            if self._file is not None:
                self._file.close()
                self._file = None
                self._writer = None

    # -- data --

    def write_snapshot(self, snapshot: MemorySnapshot) -> None:
        """Append rows for every (device, process) pair in *snapshot*."""
        with self._lock:
            if self._writer is None or self._file is None:
                return
            for device in snapshot.devices:
                if not device.processes:
                    # Still log the GPU even if no processes are visible.
                    self._writer.writerow(
                        {
                            "wall_time": snapshot.wall_time,
                            "gpu_index": device.index,
                            "gpu_name": device.name,
                            "gpu_used_bytes": device.used_memory_bytes,
                            "gpu_total_bytes": device.total_memory_bytes,
                            "gpu_util_percent": device.gpu_util_percent,
                            "pid": "",
                            "process_name": "",
                            "process_vram_bytes": "",
                            "process_type": "",
                        }
                    )
                for proc in device.processes:
                    self._writer.writerow(
                        {
                            "wall_time": snapshot.wall_time,
                            "gpu_index": device.index,
                            "gpu_name": device.name,
                            "gpu_used_bytes": device.used_memory_bytes,
                            "gpu_total_bytes": device.total_memory_bytes,
                            "gpu_util_percent": device.gpu_util_percent,
                            "pid": proc.identity.pid,
                            "process_name": proc.name,
                            "process_vram_bytes": proc.used_memory_bytes,
                            "process_type": proc.process_type,
                        }
                    )
            self._file.flush()
