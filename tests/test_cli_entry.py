"""Tests for :mod:`taskclf.cli.entry` fast-path and delegation."""

from __future__ import annotations

from unittest.mock import patch


def test_cli_entry_version_only_when_first_arg() -> None:
    """``-v`` / ``--version`` only short-circuit when argv[1] is the flag."""
    from taskclf.cli import entry as entry_mod

    with patch.object(
        entry_mod.sys,
        "argv",
        ["taskclf", "--version"],
    ):
        with patch("importlib.metadata.version", return_value="9.9.9"):
            with patch("builtins.print") as print_mock:
                entry_mod.cli_entry()
    printed = [c[0][0] for c in print_mock.call_args_list]
    assert any("9.9.9" in str(p) for p in printed)


def test_cli_entry_does_not_version_when_v_is_later_argv() -> None:
    """A later ``-v`` token (e.g. title salt) must not trigger version path."""
    from taskclf.cli import entry as entry_mod

    with patch.object(
        entry_mod.sys,
        "argv",
        [
            "entry",
            "tray",
            "--title-salt",
            "-v",
        ],
    ):
        with patch("taskclf.cli.main.cli_main") as cli_main_mock:
            entry_mod.cli_entry()
    cli_main_mock.assert_called_once()


def test_cli_entry_delegates_tray_subcommand() -> None:
    from taskclf.cli import entry as entry_mod

    with patch.object(
        entry_mod.sys,
        "argv",
        ["entry", "tray", "--help"],
    ):
        with patch("taskclf.cli.main.cli_main") as cli_main_mock:
            entry_mod.cli_entry()
    cli_main_mock.assert_called_once()
