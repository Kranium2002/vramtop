"""Export module — CSV logger."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from vramtop.backends.base import MemorySnapshot
    from vramtop.config import ExportConfig

from vramtop.export.csv_logger import CSVLogger


class ExportManager:
    """Manages export backends.

    Currently supports CSV logging.  Call ``update_snapshot`` from the
    poll loop to push data to all active exporters.
    """

    def __init__(self, config: ExportConfig, csv_path: Path | None = None) -> None:
        self._config = config
        self._csv: CSVLogger | None = None
        if csv_path is not None:
            self._csv = CSVLogger(csv_path)

    def start(self) -> None:
        if self._csv is not None:
            self._csv.start()

    def stop(self) -> None:
        if self._csv is not None:
            self._csv.stop()

    def update_snapshot(self, snapshot: MemorySnapshot) -> None:
        if self._csv is not None:
            self._csv.write_snapshot(snapshot)
