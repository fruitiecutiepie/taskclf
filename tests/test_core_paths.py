"""Tests for taskclf.core.paths — TASKCLF_HOME resolution and directory bootstrap."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from taskclf.core.paths import _SUBDIRS, ensure_taskclf_dirs, taskclf_home


class TestTaskclfHomeEnvVar:
    """TASKCLF_HOME env var takes highest priority."""

    def test_uses_env_var_when_set(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        target = tmp_path / "custom"
        monkeypatch.setenv("TASKCLF_HOME", str(target))
        assert taskclf_home() == target

    def test_expands_tilde_in_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TASKCLF_HOME", "~/my-taskclf")
        result = taskclf_home()
        assert result.is_absolute()
        assert "~" not in str(result)
        assert result.name == "my-taskclf"

    def test_resolves_relative_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TASKCLF_HOME", "relative/path")
        result = taskclf_home()
        assert result.is_absolute()


class TestTaskclfHomeMacOS:
    """Platform default on macOS."""

    def test_default_macos_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("TASKCLF_HOME", raising=False)
        with patch("taskclf.core.paths.sys") as mock_sys:
            mock_sys.platform = "darwin"
            result = taskclf_home()
        assert result == Path.home() / "Library" / "Application Support" / "taskclf"


class TestTaskclfHomeLinux:
    """Platform default on Linux / POSIX."""

    def test_default_linux_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("TASKCLF_HOME", raising=False)
        monkeypatch.delenv("XDG_DATA_HOME", raising=False)
        with patch("taskclf.core.paths.sys") as mock_sys:
            mock_sys.platform = "linux"
            result = taskclf_home()
        assert result == Path.home() / ".local" / "share" / "taskclf"

    def test_respects_xdg_data_home(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("TASKCLF_HOME", raising=False)
        monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
        with patch("taskclf.core.paths.sys") as mock_sys:
            mock_sys.platform = "linux"
            result = taskclf_home()
        assert result == tmp_path / "xdg" / "taskclf"


class TestTaskclfHomeWindows:
    """Platform default on Windows."""

    def test_default_windows_path_with_localappdata(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("TASKCLF_HOME", raising=False)
        monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "AppData" / "Local"))
        with patch("taskclf.core.paths.sys") as mock_sys:
            mock_sys.platform = "win32"
            result = taskclf_home()
        assert result == tmp_path / "AppData" / "Local" / "taskclf"

    def test_windows_fallback_without_localappdata(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("TASKCLF_HOME", raising=False)
        monkeypatch.delenv("LOCALAPPDATA", raising=False)
        with patch("taskclf.core.paths.sys") as mock_sys:
            mock_sys.platform = "win32"
            result = taskclf_home()
        assert result == Path.home() / "AppData" / "Local" / "taskclf"


class TestEnsureTaskclfDirs:
    """ensure_taskclf_dirs() creates the standard subdirectory tree."""

    def test_creates_all_subdirs(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TASKCLF_HOME", str(tmp_path / "home"))
        home = ensure_taskclf_dirs()
        assert home == tmp_path / "home"
        for sub in _SUBDIRS:
            assert (home / sub).is_dir(), f"Missing subdirectory: {sub}"

    def test_idempotent(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TASKCLF_HOME", str(tmp_path / "home"))
        ensure_taskclf_dirs()
        ensure_taskclf_dirs()
        for sub in _SUBDIRS:
            assert (tmp_path / "home" / sub).is_dir()

    def test_logs_on_creation(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setenv("TASKCLF_HOME", str(tmp_path / "home"))
        import logging

        with caplog.at_level(logging.INFO, logger="taskclf.core.paths"):
            ensure_taskclf_dirs()
        assert any("Created directory" in msg for msg in caplog.messages)

    def test_no_log_on_second_call(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setenv("TASKCLF_HOME", str(tmp_path / "home"))
        ensure_taskclf_dirs()
        import logging

        caplog.clear()
        with caplog.at_level(logging.INFO, logger="taskclf.core.paths"):
            ensure_taskclf_dirs()
        assert not any("Created directory" in msg for msg in caplog.messages)

    def test_returns_home_path(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TASKCLF_HOME", str(tmp_path / "home"))
        result = ensure_taskclf_dirs()
        assert result == tmp_path / "home"


class TestSubdirsList:
    """Verify the expected subdirectories are in _SUBDIRS."""

    def test_contains_expected_entries(self) -> None:
        expected = {"data/raw/aw", "data/processed", "models", "artifacts/telemetry", "configs", "logs"}
        assert set(_SUBDIRS) == expected


class TestCLICallbackCreatesDirectories:
    """The CLI entrypoint calls ensure_taskclf_dirs() on every invocation."""

    def test_cli_creates_home_dirs(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from typer.testing import CliRunner
        from taskclf.cli.main import app

        monkeypatch.setenv("TASKCLF_HOME", str(tmp_path / "home"))
        runner = CliRunner()
        result = runner.invoke(app, [
            "train", "list", "--models-dir", str(tmp_path / "home" / "models"),
        ])
        assert result.exit_code == 0, result.output
        for sub in _SUBDIRS:
            assert (tmp_path / "home" / sub).is_dir(), f"Missing subdirectory after CLI invoke: {sub}"
