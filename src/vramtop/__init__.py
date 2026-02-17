"""vramtop — The htop for GPU memory."""

from __future__ import annotations

from typing import Any

__version__ = "0.1.0"


def report(**kwargs: Any) -> None:
    """Start deep mode reporter (convenience wrapper).

    Usage::

        import vramtop
        vramtop.report()
    """
    from vramtop.reporter.pytorch import report as _report

    _report(**kwargs)
