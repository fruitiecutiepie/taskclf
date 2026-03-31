#!/usr/bin/env python3
"""Print the LLVM-style host target triple for this machine.

Used for payload zip names (payload-<triple>.zip) and manifest.json platform keys.
Must stay in sync with hostTargetTriple() in electron/updater.ts.
"""

from __future__ import annotations

import platform
import sys


def host_target_triple() -> str:
    machine = platform.machine().lower()
    sysname = sys.platform

    if sysname == "darwin":
        if machine in ("arm64", "aarch64"):
            return "aarch64-apple-darwin"
        if machine in ("x86_64", "amd64"):
            return "x86_64-apple-darwin"
        raise RuntimeError(f"unsupported macOS machine: {machine!r}")

    if sysname == "linux":
        if machine in ("x86_64", "amd64"):
            return "x86_64-unknown-linux-gnu"
        if machine in ("aarch64", "arm64"):
            return "aarch64-unknown-linux-gnu"
        if machine in ("i686", "i386", "x86"):
            return "i686-unknown-linux-gnu"
        raise RuntimeError(f"unsupported Linux machine: {machine!r}")

    if sysname == "win32":
        if machine in ("amd64", "x86_64"):
            return "x86_64-pc-windows-msvc"
        if machine in ("arm64", "aarch64"):
            return "aarch64-pc-windows-msvc"
        if machine in ("i386", "i686", "x86"):
            return "i686-pc-windows-msvc"
        raise RuntimeError(f"unsupported Windows machine: {machine!r}")

    raise RuntimeError(f"unsupported platform: {sysname!r}")


def main() -> None:
    print(host_target_triple())


if __name__ == "__main__":
    main()
