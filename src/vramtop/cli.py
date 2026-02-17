"""CLI entry point for vramtop."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from vramtop import __version__


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vramtop",
        description="The htop for GPU memory. NVIDIA GPU monitor.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"vramtop {__version__}",
    )
    parser.add_argument(
        "--no-kill",
        action="store_true",
        default=False,
        help="Disable process kill functionality",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        metavar="PATH",
        help="Path to config TOML file",
    )
    parser.add_argument(
        "--refresh-rate",
        type=float,
        default=None,
        metavar="FLOAT",
        help="Refresh rate in seconds (0.5-10)",
    )
    parser.add_argument(
        "--accessible",
        action="store_true",
        default=False,
        help="Enable accessible mode (text labels alongside symbols, NO_COLOR)",
    )
    parser.add_argument(
        "--export-csv",
        type=Path,
        default=None,
        metavar="FILE",
        help="Log GPU/process data to a CSV file",
    )
    return parser


def main() -> None:
    """Entry point for the vramtop CLI."""
    parser = _build_parser()
    args = parser.parse_args()

    from vramtop.backends import get_backend
    from vramtop.backends.base import BackendError
    from vramtop.config import ConfigHolder

    config_holder = ConfigHolder(path=args.config)

    # Override config with CLI args
    if args.no_kill:
        config_holder.config.display.enable_kill = False
    if args.refresh_rate is not None:
        config_holder.config.general.refresh_rate = args.refresh_rate

    config_holder.install_signal_handler()

    backend = get_backend()
    try:
        backend.initialize()
    except BackendError as exc:
        print(
            f"Error: {exc}\n\n"
            "vramtop requires an NVIDIA GPU with drivers installed.\n"
            "Make sure nvidia-smi works before running vramtop.",
            file=sys.stderr,
        )
        raise SystemExit(1) from None

    # CSV export (optional)
    csv_path: Path | None = args.export_csv

    from vramtop.ui.app import VramtopApp

    app = VramtopApp(
        backend=backend,
        config_holder=config_holder,
        accessible=args.accessible,
        csv_path=csv_path,
    )
    app.run()
