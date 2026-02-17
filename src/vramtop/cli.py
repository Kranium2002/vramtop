"""CLI entry point for vramtop."""

from __future__ import annotations

import argparse
import os
import subprocess
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

    subparsers = parser.add_subparsers(dest="command")

    # Monitor subcommand (default behavior)
    monitor = subparsers.add_parser(
        "monitor", help="Launch the GPU monitoring TUI (default)"
    )
    _add_monitor_args(monitor)

    # Wrap subcommand
    wrap = subparsers.add_parser(
        "wrap",
        help="Run a command with deep mode reporting enabled",
    )
    wrap.add_argument(
        "wrapped_cmd",
        nargs=argparse.REMAINDER,
        help="Command to run with VRAMTOP_REPORT=1",
    )

    # Add monitor args to the top-level parser too for backward compat
    _add_monitor_args(parser)

    return parser


def _add_monitor_args(parser: argparse.ArgumentParser) -> None:
    """Add monitor-mode arguments to a parser."""
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


def _run_monitor(args: argparse.Namespace) -> None:
    """Run the TUI monitor."""
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


def _run_wrap(args: argparse.Namespace) -> None:
    """Run a command with VRAMTOP_REPORT=1 set."""
    cmd: list[str] = args.wrapped_cmd
    if not cmd:
        print("Error: wrap requires a command to run.", file=sys.stderr)
        raise SystemExit(1)

    # Strip leading '--' if present
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]

    if not cmd:
        print("Error: wrap requires a command to run.", file=sys.stderr)
        raise SystemExit(1)

    env = os.environ.copy()
    env["VRAMTOP_REPORT"] = "1"

    result = subprocess.run(cmd, env=env)
    raise SystemExit(result.returncode)


def main() -> None:
    """Entry point for the vramtop CLI."""
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "wrap":
        _run_wrap(args)
    else:
        # Default to monitor (both "monitor" subcommand and no subcommand)
        _run_monitor(args)
