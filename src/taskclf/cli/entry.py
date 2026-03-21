"""Lightweight CLI entry point for fast --version response.

This module is the console-script target.  It intercepts ``-v`` /
``--version`` *before* importing the full command tree in
:mod:`taskclf.cli.main`, avoiding the cost of parsing 3 000+ lines
of Typer definitions and their transitive dependencies when the user
only wants the version string.
"""

from __future__ import annotations

import sys


def cli_entry() -> None:
    """Console-script entry point.

    Fast-paths ``--version`` / ``-v``; delegates everything else to
    :func:`taskclf.cli.main.cli_main` (which includes the crash handler).
    """
    if "-v" in sys.argv[1:] or "--version" in sys.argv[1:]:
        from importlib.metadata import version

        print(f"taskclf {version('taskclf')}")
        return

    from taskclf.cli.main import cli_main

    cli_main()
