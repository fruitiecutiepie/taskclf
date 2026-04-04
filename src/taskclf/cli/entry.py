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

    Only the first argument after the executable is checked for ``-v`` /
    ``--version``. That avoids treating a ``-v`` value later in the argv list
    (e.g. ``--title-salt -v``) as a version-only invocation.
    """
    if len(sys.argv) >= 2 and sys.argv[1] in ("-v", "--version"):
        from importlib.metadata import version

        print(f"taskclf {version('taskclf')}")
        return

    from taskclf.cli.main import cli_main

    cli_main()


if __name__ == "__main__":
    cli_entry()
