"""Tests for PyInstaller payload staging (Electron sidecar contract)."""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

import pytest

# scripts/ is not a package; load helpers for tests.
_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = _ROOT / "scripts"


@pytest.fixture
def payload_build():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "payload_build",
        _SCRIPTS / "payload_build.py",
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def host_target_triple_mod():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "host_target_triple",
        _SCRIPTS / "host_target_triple.py",
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_host_target_triple_is_in_electron_supported_set(
    payload_build,
    host_target_triple_mod,
) -> None:
    """Runtime triple must be a key the Electron updater can select from manifest.json."""
    triple = host_target_triple_mod.host_target_triple()
    assert triple in payload_build.ELECTRON_SUPPORTED_TRIPLES


def test_stage_onedir_to_backend_copies_tree(payload_build, tmp_path: Path) -> None:
    onedir = tmp_path / "entry"
    onedir.mkdir()
    (onedir / "_internal").mkdir()
    (onedir / "_internal" / "lib.txt").write_text("x", encoding="utf-8")
    exe_name = "entry.exe" if sys.platform == "win32" else "entry"
    (onedir / exe_name).write_bytes(b"\x00")

    backend = tmp_path / "backend"
    payload_build.stage_onedir_to_backend(onedir, backend)

    assert (backend / exe_name).is_file()
    assert (backend / "_internal" / "lib.txt").read_text(encoding="utf-8") == "x"


def test_zip_payload_directory_preserves_backend_prefix(
    payload_build, tmp_path: Path
) -> None:
    payload_root = tmp_path / "payload"
    (payload_root / "backend").mkdir(parents=True)
    exe_name = "entry.exe" if sys.platform == "win32" else "entry"
    (payload_root / "backend" / exe_name).write_text("exe", encoding="utf-8")

    zip_path = tmp_path / "out.zip"
    payload_build.zip_payload_directory(payload_root, zip_path)

    with zipfile.ZipFile(zip_path) as zf:
        names = sorted(zf.namelist())
    assert f"backend/{exe_name}" in names


def _minimal_repo_with_static(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    static = repo / "src" / "taskclf" / "ui" / "static"
    static.mkdir(parents=True)
    (static / "index.html").write_text("<!doctype html><html></html>", encoding="utf-8")
    return repo


def test_pyinstaller_argv_does_not_collect_all_taskclf(
    payload_build,
    tmp_path: Path,
) -> None:
    """Regression: ``--collect-all taskclf`` sweeps frontend/node_modules into the bundle."""
    repo = _minimal_repo_with_static(tmp_path)
    argv = payload_build.pyinstaller_argv(repo)
    pairs = list(zip(argv, argv[1:]))
    assert ("--collect-all", "taskclf") not in pairs


def test_pyinstaller_argv_collects_taskclf_submodules_only(
    payload_build,
    tmp_path: Path,
) -> None:
    repo = _minimal_repo_with_static(tmp_path)
    argv = payload_build.pyinstaller_argv(repo)
    assert "--collect-submodules" in argv
    idx = argv.index("--collect-submodules")
    assert argv[idx + 1] == "taskclf"


def test_pyinstaller_argv_collects_third_party_runtime_deps(
    payload_build,
    tmp_path: Path,
) -> None:
    repo = _minimal_repo_with_static(tmp_path)
    argv = payload_build.pyinstaller_argv(repo)
    for pkg in (
        "lightgbm",
        "numpy",
        "pandas",
        "sklearn",
        "tomli_w",
        "uvicorn",
        "fastapi",
    ):
        assert pkg in argv
        i = argv.index(pkg)
        assert argv[i - 1] == "--collect-all"


def test_strip_ui_frontend_dev_tree_removes_frontend(
    payload_build,
    tmp_path: Path,
) -> None:
    onedir = tmp_path / "entry"
    internal = onedir / "_internal" / "taskclf" / "ui" / "frontend"
    (internal / "node_modules" / "x").mkdir(parents=True)
    (internal / "node_modules" / "x" / "pkg.json").write_text("{}", encoding="utf-8")

    payload_build.strip_ui_frontend_dev_tree(onedir)

    assert not internal.exists()


def test_assert_no_forbidden_dev_paths_raises_if_frontend_present(
    payload_build,
    tmp_path: Path,
) -> None:
    onedir = tmp_path / "entry"
    fe = onedir / "_internal" / "taskclf" / "ui" / "frontend"
    fe.mkdir(parents=True)
    (fe / "package.json").write_text("{}", encoding="utf-8")

    with pytest.raises(RuntimeError, match="raw frontend source tree"):
        payload_build.assert_no_forbidden_dev_paths_in_onedir(onedir)
