"""Collect environment and runtime diagnostics for bug reports.

Used by both ``taskclf diagnostics`` (CLI) and the tray "Report Issue" menu
item so that users never have to gather this information manually.
"""

from __future__ import annotations

import platform
import sys
import urllib.request
from pathlib import Path

from importlib.metadata import version as _pkg_version


def collect_diagnostics(
    *,
    aw_host: str,
    data_dir: str,
    models_dir: str,
    include_logs: bool = False,
    log_lines: int = 50,
) -> dict[str, object]:
    """Gather environment and runtime info for bug reports.

    Returns a dict with sections that can be serialised to JSON or
    pretty-printed for human consumption.
    """
    from taskclf.core.config import UserConfig
    from taskclf.core.paths import taskclf_home
    from taskclf.model_registry import list_bundles

    home = taskclf_home()

    info: dict[str, object] = {}

    try:
        info["taskclf_version"] = _pkg_version("taskclf")
    except Exception:
        info["taskclf_version"] = "unknown"

    info["python_version"] = sys.version
    info["os"] = platform.platform()
    info["architecture"] = platform.machine()
    info["taskclf_home"] = str(home)

    # -- ActivityWatch reachability ---
    aw_url = f"{aw_host.rstrip('/')}/api/0/info"
    try:
        with urllib.request.urlopen(aw_url, timeout=5) as resp:
            info["activitywatch"] = {"reachable": True, "status": resp.status}
    except Exception as exc:
        info["activitywatch"] = {"reachable": False, "error": str(exc)}

    # -- Model bundles ---
    bundles = list_bundles(Path(models_dir))
    info["model_bundles"] = [
        {
            "model_id": b.model_id,
            "valid": b.valid,
            "created_at": b.created_at.isoformat() if b.created_at else None,
        }
        for b in bundles
    ]

    # -- Config (redact user_id) ---
    try:
        cfg = UserConfig(data_dir).as_dict()
        cfg["user_id"] = "[REDACTED]"
        info["config"] = cfg
    except Exception as exc:
        info["config"] = {"error": str(exc)}

    # -- Disk usage ---
    def _dir_size(p: Path) -> int | None:
        if not p.is_dir():
            return None
        return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())

    info["disk_usage"] = {
        "data": _dir_size(home / "data"),
        "models": _dir_size(home / "models"),
        "logs": _dir_size(home / "logs"),
    }

    # -- Log tail ---
    if include_logs:
        log_file = home / "logs" / "taskclf.log"
        if log_file.is_file():
            try:
                all_lines = log_file.read_text("utf-8").splitlines()
                info["log_tail"] = all_lines[-log_lines:]
            except Exception:
                info["log_tail"] = ["<unable to read log file>"]
        else:
            info["log_tail"] = ["<log file not found>"]

    return info


def format_diagnostics_text(info: dict[str, object]) -> str:
    """Render diagnostics dict as human-readable text."""
    lines: list[str] = []
    lines.append("taskclf diagnostics")
    lines.append("=" * 40)
    lines.append(f"  version      : {info.get('taskclf_version', '?')}")
    lines.append(f"  python       : {info.get('python_version', '?')}")
    lines.append(f"  os           : {info.get('os', '?')}")
    lines.append(f"  architecture : {info.get('architecture', '?')}")
    lines.append(f"  taskclf_home : {info.get('taskclf_home', '?')}")

    aw = info.get("activitywatch", {})
    if isinstance(aw, dict):
        if aw.get("reachable"):
            lines.append(f"  activitywatch: reachable (HTTP {aw.get('status', '?')})")
        else:
            lines.append(f"  activitywatch: unreachable ({aw.get('error', '?')})")

    lines.append("")
    lines.append("model bundles")
    lines.append("-" * 40)
    bundles = info.get("model_bundles", [])
    if isinstance(bundles, list) and bundles:
        for b in bundles:
            if isinstance(b, dict):
                tag = "valid" if b.get("valid") else "INVALID"
                lines.append(f"  {b.get('model_id', '?')} [{tag}] created={b.get('created_at', '?')}")
    else:
        lines.append("  (none)")

    lines.append("")
    lines.append("config")
    lines.append("-" * 40)
    cfg = info.get("config", {})
    if isinstance(cfg, dict):
        for k, v in cfg.items():
            lines.append(f"  {k}: {v}")

    lines.append("")
    lines.append("disk usage")
    lines.append("-" * 40)
    disk = info.get("disk_usage", {})
    if isinstance(disk, dict):
        for k, v in disk.items():
            if v is None:
                lines.append(f"  {k}: (not found)")
            elif isinstance(v, (int, float)):
                mb = v / (1024 * 1024)
                lines.append(f"  {k}: {mb:.2f} MB")
            else:
                lines.append(f"  {k}: {v}")

    log_tail = info.get("log_tail")
    if isinstance(log_tail, list):
        lines.append("")
        lines.append(f"log tail (last {len(log_tail)} lines)")
        lines.append("-" * 40)
        lines.extend(f"  {line}" for line in log_tail)

    lines.append("")
    return "\n".join(lines)
