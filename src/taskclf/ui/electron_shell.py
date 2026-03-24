"""Electron desktop shell launcher for taskclf.

This module is the boundary between the Python CLI and the Electron
desktop shell. It prepares environment variables for the Electron main
process, which in turn spawns the existing tray backend in browser mode
without opening a separate browser window.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from taskclf.core.defaults import (
    DEFAULT_AW_HOST,
    DEFAULT_DATA_DIR,
    DEFAULT_MODELS_DIR,
    DEFAULT_POLL_SECONDS,
    DEFAULT_TITLE_SALT,
    DEFAULT_TRANSITION_MINUTES,
)


@dataclass(frozen=True, slots=True)
class ElectronLaunchConfig:
    """Parameters forwarded from the Python CLI to the Electron shell."""

    model_dir: Path | None = None
    models_dir: Path = Path(DEFAULT_MODELS_DIR)
    aw_host: str = DEFAULT_AW_HOST
    poll_seconds: int = DEFAULT_POLL_SECONDS
    title_salt: str = DEFAULT_TITLE_SALT
    data_dir: Path = Path(DEFAULT_DATA_DIR)
    transition_minutes: int = DEFAULT_TRANSITION_MINUTES
    ui_port: int = 8741
    dev: bool = False
    username: str | None = None
    retrain_config: Path | None = None


def _repo_root_resolve() -> Path:
    return Path(__file__).resolve().parents[3]


def _electron_dir_resolve() -> Path:
    return _repo_root_resolve() / "electron"


def _frontend_dir_resolve() -> Path:
    return _repo_root_resolve() / "src" / "taskclf" / "ui" / "frontend"


def _frontend_static_dir_resolve() -> Path:
    return _repo_root_resolve() / "src" / "taskclf" / "ui" / "static"


def _pnpm_binary_resolve() -> str:
    pnpm = shutil.which("pnpm")
    if pnpm is None:
        raise RuntimeError("pnpm is required to launch the Electron shell")
    return pnpm


def _electron_env_build(
    config: ElectronLaunchConfig, *, python_executable: str | None = None
) -> dict[str, str]:
    env = os.environ.copy()
    env["TASKCLF_ELECTRON_PYTHON_EXECUTABLE"] = python_executable or sys.executable
    env["TASKCLF_ELECTRON_AW_HOST"] = config.aw_host
    env["TASKCLF_ELECTRON_POLL_SECONDS"] = str(config.poll_seconds)
    env["TASKCLF_ELECTRON_TITLE_SALT"] = config.title_salt
    env["TASKCLF_ELECTRON_DATA_DIR"] = str(config.data_dir)
    env["TASKCLF_ELECTRON_TRANSITION_MINUTES"] = str(config.transition_minutes)
    env["TASKCLF_ELECTRON_UI_PORT"] = str(config.ui_port)
    env["TASKCLF_ELECTRON_DEV"] = "1" if config.dev else "0"
    env["TASKCLF_ELECTRON_MODELS_DIR"] = str(config.models_dir)
    if config.model_dir is not None:
        env["TASKCLF_ELECTRON_MODEL_DIR"] = str(config.model_dir)
    if config.username is not None:
        env["TASKCLF_ELECTRON_USERNAME"] = config.username
    if config.retrain_config is not None:
        env["TASKCLF_ELECTRON_RETRAIN_CONFIG"] = str(config.retrain_config)
    return env


def _path_latest_mtime_ns(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_mtime_ns

    latest = 0
    for child in path.rglob("*"):
        if child.is_file():
            latest = max(latest, child.stat().st_mtime_ns)
    return latest


def _frontend_build_needed(frontend_dir: Path, static_dir: Path) -> bool:
    index_path = static_dir / "index.html"
    if not index_path.is_file():
        return True

    source_mtime = max(
        _path_latest_mtime_ns(frontend_dir / "src"),
        _path_latest_mtime_ns(frontend_dir / "public"),
        _path_latest_mtime_ns(frontend_dir / "package.json"),
        _path_latest_mtime_ns(frontend_dir / "vite.config.ts"),
        _path_latest_mtime_ns(frontend_dir / "tsconfig.json"),
    )
    built_mtime = _path_latest_mtime_ns(static_dir)
    return source_mtime > built_mtime


def _frontend_prepare(*, dev: bool) -> None:
    frontend_dir = _frontend_dir_resolve()
    if not frontend_dir.is_dir():
        raise RuntimeError("Frontend sources were not found. Run from a repo checkout.")

    pnpm = _pnpm_binary_resolve()
    if not (frontend_dir / "node_modules").is_dir():
        subprocess.run(
            [pnpm, "install", "--frozen-lockfile"],
            cwd=frontend_dir,
            check=True,
        )

    if dev:
        return

    static_dir = _frontend_static_dir_resolve()
    if _frontend_build_needed(frontend_dir, static_dir):
        subprocess.run(
            [pnpm, "run", "build"],
            cwd=frontend_dir,
            check=True,
        )


def launch_electron_shell(
    config: ElectronLaunchConfig, *, python_executable: str | None = None
) -> None:
    """Build the Electron shell and run it in the foreground."""

    electron_dir = _electron_dir_resolve()
    if not electron_dir.is_dir():
        raise RuntimeError(
            "Electron shell sources were not found. Run from a repo checkout."
        )
    if not (electron_dir / "package.json").is_file():
        raise RuntimeError(
            "electron/package.json is missing. Install the Electron shell sources first."
        )
    if not (electron_dir / "node_modules").is_dir():
        raise RuntimeError(
            "Electron dependencies are missing. Run `pnpm install` in `electron/`."
        )

    _frontend_prepare(dev=config.dev)
    env = _electron_env_build(config, python_executable=python_executable)
    subprocess.run(
        [_pnpm_binary_resolve(), "run", "start"],
        cwd=electron_dir,
        env=env,
        check=True,
    )
