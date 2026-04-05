#!/usr/bin/env python3
"""Build a PyInstaller one-folder sidecar and stage it for Electron payload releases.

The Electron updater expects ``payload-<triple>.zip`` to extract to a tree containing
``backend/entry`` (Unix) or ``backend/entry.exe`` (Windows). This script runs
PyInstaller ``--onedir --name entry``, copies the output into ``build/payload/backend/``,
and writes ``build/payload-<triple>.zip``.

Must stay aligned with ``hostTargetTriple()`` in ``electron/updater.ts`` for triple
strings used in filenames and ``manifest.json``.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

# LLVM-style triples the Electron updater can select (see electron/updater.ts).
# Used by tests to keep Python triple logic in sync with the shell.
ELECTRON_SUPPORTED_TRIPLES: frozenset[str] = frozenset(
    {
        "aarch64-apple-darwin",
        "x86_64-apple-darwin",
        "x86_64-unknown-linux-gnu",
        "aarch64-unknown-linux-gnu",
        "i686-unknown-linux-gnu",
        "x86_64-pc-windows-msvc",
        "aarch64-pc-windows-msvc",
        "i686-pc-windows-msvc",
    }
)


def repo_root_from_here() -> Path:
    return Path(__file__).resolve().parent.parent


def add_data_arg(static_dir: Path) -> str:
    """PyInstaller --add-data uses os.pathsep between source and dest."""
    return f"{static_dir}{os.pathsep}taskclf/ui/static"


def pyinstaller_onedir_dir(distpath: Path) -> Path:
    """Directory PyInstaller creates for ``--name entry --onedir``."""
    return distpath / "entry"


# Dev-only tree under ``src/taskclf/ui/frontend`` (SolidJS sources, node_modules).
# Must never ship in the Electron sidecar; runtime UI is ``taskclf/ui/static`` only.
FORBIDDEN_ONEDIR_REL_PATHS: tuple[str, ...] = (
    "taskclf/ui/frontend/node_modules",
    "taskclf/ui/frontend/.pnpm",
)


def pyinstaller_argv(repo_root: Path) -> list[str]:
    """Arguments for ``python -m PyInstaller`` (excluding the interpreter)."""
    src = repo_root / "src"
    entry_script = src / "taskclf" / "cli" / "entry.py"
    static_dir = src / "taskclf" / "ui" / "static"
    distpath = repo_root / "build" / "pyinstaller" / "dist"
    workpath = repo_root / "build" / "pyinstaller" / "build"

    if not static_dir.is_dir():
        msg = f"Missing built UI static dir: {static_dir}. Run `make ui-build` first."
        raise FileNotFoundError(msg)

    # Do not use ``--collect-all taskclf``: it sweeps the entire source tree under
    # ``src/taskclf``, including ``ui/frontend/node_modules`` (~hundreds of MB).
    # Collect Python submodules only; ship the web UI via ``--add-data`` → static.
    # Heavy third-party deps still need ``--collect-all`` for native libs / metadata.
    collect_all_packages = (
        "lightgbm",
        "sklearn",
        "scipy",
        "matplotlib",
        "pandas",
        "numpy",
        "pyarrow",
        "duckdb",
        "PIL",
        "pystray",
        "tomli_w",
        "uvicorn",
        "fastapi",
        "pydantic",
    )

    args: list[str] = [
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--onedir",
        "--name",
        "entry",
        "--distpath",
        str(distpath),
        "--workpath",
        str(workpath),
        "--paths",
        str(src),
        "--add-data",
        add_data_arg(static_dir),
        "--collect-submodules",
        "taskclf",
    ]
    for pkg in collect_all_packages:
        args.extend(("--collect-all", pkg))
    args.extend(
        (
            "--hidden-import",
            "taskclf.cli.main",
            "--hidden-import",
            "uvicorn.logging",
            "--hidden-import",
            "uvicorn.loops",
            "--hidden-import",
            "uvicorn.loops.auto",
            "--hidden-import",
            "uvicorn.protocols",
            "--hidden-import",
            "uvicorn.protocols.http",
            "--hidden-import",
            "uvicorn.protocols.http.auto",
            "--hidden-import",
            "uvicorn.protocols.websockets",
            "--hidden-import",
            "uvicorn.protocols.websockets.auto",
            str(entry_script),
        )
    )
    return args


def strip_ui_frontend_dev_tree(onedir: Path) -> None:
    """Remove SolidJS sources and node_modules from the one-folder output.

    The shipped UI is only ``taskclf/ui/static`` (via ``--add-data``). Dev mode
    may reference ``taskclf/ui/frontend`` at runtime from a repo checkout; that
    tree must never appear in the PyInstaller bundle.
    """
    internal = onedir / "_internal"
    if not internal.is_dir():
        return
    frontend = internal / "taskclf" / "ui" / "frontend"
    if frontend.is_dir():
        shutil.rmtree(frontend)


def assert_no_forbidden_dev_paths_in_onedir(onedir: Path) -> None:
    """Fail the build if dev-only paths leaked into ``_internal``."""
    internal = onedir / "_internal"
    if not internal.is_dir():
        return
    frontend_root = internal / "taskclf" / "ui" / "frontend"
    if frontend_root.exists():
        msg = (
            "PyInstaller bundle must not include the raw frontend source tree "
            f"(found {frontend_root}). Use taskclf/ui/static only."
        )
        raise RuntimeError(msg)
    for rel in FORBIDDEN_ONEDIR_REL_PATHS:
        path = internal / rel
        if path.exists():
            msg = (
                "PyInstaller bundle must not include dev frontend paths "
                f"(found {path}). Check payload_build.py collect rules."
            )
            raise RuntimeError(msg)


def run_pyinstaller(repo_root: Path) -> Path:
    argv = pyinstaller_argv(repo_root)
    subprocess.run([sys.executable, *argv], cwd=repo_root, check=True)
    distpath = repo_root / "build" / "pyinstaller" / "dist"
    onedir = pyinstaller_onedir_dir(distpath)
    if not onedir.is_dir():
        msg = f"Expected PyInstaller output directory missing: {onedir}"
        raise FileNotFoundError(msg)
    strip_ui_frontend_dev_tree(onedir)
    assert_no_forbidden_dev_paths_in_onedir(onedir)
    return onedir


def stage_onedir_to_backend(onedir: Path, backend: Path) -> None:
    """Copy a PyInstaller one-folder tree into ``backend`` (Electron sidecar root)."""
    if backend.exists():
        shutil.rmtree(backend)
    shutil.copytree(onedir, backend)


def zip_payload_directory(payload_root: Path, zip_path: Path) -> None:
    """Zip ``payload_root`` so archive paths are relative to ``payload_root`` (e.g. ``backend/...``)."""
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in payload_root.rglob("*"):
            if path.is_file():
                arc = path.relative_to(payload_root)
                zf.write(path, arc)


def build_payload(repo_root: Path, triple: str) -> Path:
    """Run PyInstaller, stage under ``build/payload/backend``, write ``build/payload-<triple>.zip``."""
    onedir = run_pyinstaller(repo_root)
    payload_root = repo_root / "build" / "payload"
    backend = payload_root / "backend"
    payload_root.mkdir(parents=True, exist_ok=True)
    stage_onedir_to_backend(onedir, backend)

    # Executable name must match electron/updater.ts (entry / entry.exe).
    exe = backend / ("entry.exe" if sys.platform == "win32" else "entry")
    if not exe.is_file():
        msg = f"Expected sidecar executable missing after staging: {exe}"
        raise FileNotFoundError(msg)

    zip_path = repo_root / "build" / f"payload-{triple}.zip"
    zip_payload_directory(payload_root, zip_path)
    return zip_path


def _load_host_target_triple(repo_root: Path):
    """Load ``host_target_triple`` without requiring ``scripts`` to be a package."""
    path = repo_root / "scripts" / "host_target_triple.py"
    spec = importlib.util.spec_from_file_location("host_target_triple", path)
    if spec is None or spec.loader is None:
        msg = f"Cannot load {path}"
        raise ImportError(msg)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.host_target_triple


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build PyInstaller payload zip for Electron sidecar."
    )
    parser.add_argument(
        "--triple",
        default="",
        help="LLVM-style triple for the zip filename (default: host_target_triple.py)",
    )
    args = parser.parse_args()
    root = repo_root_from_here()
    if args.triple:
        triple = args.triple
    else:
        triple = _load_host_target_triple(root)()

    zip_path = build_payload(root, triple)
    print(f"Payload built at {zip_path}")


if __name__ == "__main__":
    main()
