"""Tests for the Electron desktop shell launcher."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from taskclf.ui.electron_shell import ElectronLaunchConfig, launch_electron_shell


class TestLaunchElectronShell:
    def test_requires_installed_node_modules(self, tmp_path: Path) -> None:
        electron_dir = tmp_path / "electron"
        electron_dir.mkdir()
        (electron_dir / "package.json").write_text("{}")

        with patch(
            "taskclf.ui.electron_shell._electron_dir_resolve", return_value=electron_dir
        ):
            with pytest.raises(RuntimeError, match="pnpm install"):
                launch_electron_shell(ElectronLaunchConfig(data_dir=tmp_path / "data"))

    @patch("taskclf.ui.electron_shell.subprocess.run")
    @patch("taskclf.ui.electron_shell._pnpm_binary_resolve", return_value="pnpm")
    @patch("taskclf.ui.electron_shell._frontend_prepare")
    def test_runs_electron_with_forwarded_env(
        self,
        mock_frontend_prepare: MagicMock,
        _mock_pnpm: MagicMock,
        mock_run: MagicMock,
        tmp_path: Path,
    ) -> None:
        electron_dir = tmp_path / "electron"
        node_modules = electron_dir / "node_modules"
        electron_dir.mkdir()
        node_modules.mkdir()
        (electron_dir / "package.json").write_text("{}")

        config = ElectronLaunchConfig(
            model_dir=tmp_path / "models" / "run_001",
            models_dir=tmp_path / "models",
            aw_host="http://localhost:5600",
            poll_seconds=15,
            title_salt="salt-123",
            data_dir=tmp_path / "data",
            transition_minutes=7,
            ui_port=9123,
            dev=True,
            username="Audrey",
            retrain_config=tmp_path / "configs" / "retrain.yaml",
        )

        with patch(
            "taskclf.ui.electron_shell._electron_dir_resolve", return_value=electron_dir
        ):
            launch_electron_shell(config, python_executable="/tmp/.venv/bin/python")

        mock_frontend_prepare.assert_called_once_with(dev=True)
        mock_run.assert_called_once()
        args, kwargs = mock_run.call_args
        assert args[0] == ["pnpm", "run", "start"]
        assert kwargs["cwd"] == electron_dir
        assert kwargs["check"] is True
        env = kwargs["env"]
        assert env["TASKCLF_ELECTRON_PYTHON_EXECUTABLE"] == "/tmp/.venv/bin/python"
        assert env["TASKCLF_ELECTRON_MODEL_DIR"] == str(config.model_dir)
        assert env["TASKCLF_ELECTRON_MODELS_DIR"] == str(config.models_dir)
        assert env["TASKCLF_ELECTRON_AW_HOST"] == "http://localhost:5600"
        assert env["TASKCLF_ELECTRON_POLL_SECONDS"] == "15"
        assert env["TASKCLF_ELECTRON_TITLE_SALT"] == "salt-123"
        assert env["TASKCLF_ELECTRON_DATA_DIR"] == str(config.data_dir)
        assert env["TASKCLF_ELECTRON_TRANSITION_MINUTES"] == "7"
        assert env["TASKCLF_ELECTRON_UI_PORT"] == "9123"
        assert env["TASKCLF_ELECTRON_DEV"] == "1"
        assert env["TASKCLF_ELECTRON_USERNAME"] == "Audrey"
        assert env["TASKCLF_ELECTRON_RETRAIN_CONFIG"] == str(config.retrain_config)


class TestFrontendPrepare:
    @patch("taskclf.ui.electron_shell.subprocess.run")
    @patch("taskclf.ui.electron_shell._pnpm_binary_resolve", return_value="pnpm")
    def test_builds_static_bundle_when_stale(
        self,
        _mock_pnpm: MagicMock,
        mock_run: MagicMock,
        tmp_path: Path,
    ) -> None:
        from taskclf.ui.electron_shell import _frontend_prepare

        frontend_dir = tmp_path / "frontend"
        src_dir = frontend_dir / "src"
        public_dir = frontend_dir / "public"
        node_modules = frontend_dir / "node_modules"
        static_dir = tmp_path / "static"

        src_dir.mkdir(parents=True)
        public_dir.mkdir()
        node_modules.mkdir()
        static_dir.mkdir()

        (frontend_dir / "package.json").write_text("{}")
        (frontend_dir / "vite.config.ts").write_text("export default {};")
        (frontend_dir / "tsconfig.json").write_text("{}")
        (static_dir / "index.html").write_text("<html></html>")
        (src_dir / "App.tsx").write_text("export const App = () => null;")

        with (
            patch(
                "taskclf.ui.electron_shell._frontend_dir_resolve",
                return_value=frontend_dir,
            ),
            patch(
                "taskclf.ui.electron_shell._frontend_static_dir_resolve",
                return_value=static_dir,
            ),
        ):
            _frontend_prepare(dev=False)

        mock_run.assert_called_once_with(
            ["pnpm", "run", "build"],
            cwd=frontend_dir,
            check=True,
        )
