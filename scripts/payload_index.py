"""Build or preserve the GitHub Pages payload index.

GitHub Releases are the source of truth for ``payload-index.json``.

Current workflows regenerate a fresh index from published releases for both
payload-release and docs deploys. ``preserve-current`` remains available as an
optional/manual mode when an existing published copy should be reused.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

DEFAULT_REPO = "fruitiecutiepie/taskclf"
DEFAULT_CURRENT_INDEX_URL = (
    "https://fruitiecutiepie.github.io/taskclf/payload-index.json"
)
VERSION_TAG_RE = re.compile(r"^v(\d+)\.(\d+)\.(\d+)$")

JsonObject = dict[str, Any]


def version_sort_key(version: str) -> tuple[int, int, int]:
    match = VERSION_TAG_RE.fullmatch(f"v{version}")
    if match is None:
        raise ValueError(f"Unsupported payload version: {version}")
    major, minor, patch = match.groups()
    return int(major), int(minor), int(patch)


def payload_manifest_url(repo: str, tag: str) -> str:
    return f"https://github.com/{repo}/releases/download/{tag}/manifest.json"


def build_payload_index_from_releases(
    releases: Sequence[Mapping[str, Any]],
    repo: str,
    *,
    generated_at: dt.datetime | None = None,
) -> JsonObject:
    """Build a payload index from the published ``v*`` releases."""
    ts = generated_at or dt.datetime.now(dt.UTC)
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=dt.UTC)
    else:
        ts = ts.astimezone(dt.UTC)

    payloads: list[JsonObject] = []
    for release in releases:
        tag = str(release.get("tag_name", ""))
        if VERSION_TAG_RE.fullmatch(tag) is None:
            continue
        if bool(release.get("draft")) or bool(release.get("prerelease")):
            continue

        assets = {
            str(asset.get("name"))
            for asset in release.get("assets", [])
            if asset.get("name")
        }
        if "manifest.json" not in assets:
            continue

        payloads.append(
            {
                "version": tag[1:],
                "manifest_url": payload_manifest_url(repo, tag),
            }
        )

    payloads.sort(key=lambda item: version_sort_key(item["version"]), reverse=True)
    return {
        "kind": "payload-index",
        "schema_version": 1,
        "generated_at": ts.replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "payloads": payloads,
    }


def is_payload_index_document(payload_index: Mapping[str, Any]) -> bool:
    if payload_index.get("kind") != "payload-index":
        return False
    payloads = payload_index.get("payloads")
    if not isinstance(payloads, list):
        return False
    for entry in payloads:
        if not isinstance(entry, Mapping):
            return False
        if not isinstance(entry.get("version"), str):
            return False
        if not isinstance(entry.get("manifest_url"), str):
            return False
    return True


def choose_payload_index(
    mode: str,
    *,
    current_payload_index: Mapping[str, Any] | None,
    releases: Sequence[Mapping[str, Any]],
    repo: str,
    generated_at: dt.datetime | None = None,
) -> JsonObject:
    """Return the payload index for the requested workflow mode."""
    if mode == "preserve-current" and current_payload_index is not None:
        if is_payload_index_document(current_payload_index):
            return dict(current_payload_index)
    return build_payload_index_from_releases(
        releases,
        repo,
        generated_at=generated_at,
    )


def fetch_json(
    url: str,
    *,
    token: str = "",
    timeout: float = 15.0,
    no_cache: bool = False,
) -> Any:
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "taskclf-payload-index",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if no_cache:
        headers["Cache-Control"] = "no-cache"
        headers["Pragma"] = "no-cache"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as res:
        return json.load(res)


def fetch_releases(repo: str, *, token: str = "") -> list[JsonObject]:
    url = f"https://api.github.com/repos/{repo}/releases?per_page=100"
    data = fetch_json(url, token=token, no_cache=True)
    if not isinstance(data, list):
        raise RuntimeError("GitHub releases API returned a non-list response")
    return [dict(item) for item in data if isinstance(item, Mapping)]


def fetch_current_payload_index(
    current_url: str,
    *,
    timeout: float = 15.0,
) -> JsonObject | None:
    try:
        data = fetch_json(current_url, timeout=timeout, no_cache=True)
    except (
        urllib.error.HTTPError,
        urllib.error.URLError,
        TimeoutError,
        json.JSONDecodeError,
    ):
        return None
    if not isinstance(data, Mapping):
        return None
    if not is_payload_index_document(data):
        return None
    return dict(data)


def write_payload_index(output_path: Path, payload_index: Mapping[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(f"{json.dumps(payload_index, indent=2)}\n", encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate or preserve the GitHub Pages payload index.",
    )
    parser.add_argument(
        "mode",
        choices=("generate-from-releases", "preserve-current"),
        help="How to source payload-index.json for this workflow run.",
    )
    parser.add_argument(
        "--repo",
        default=os.environ.get("GITHUB_REPOSITORY", DEFAULT_REPO),
        help="GitHub repo in owner/name form.",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("GITHUB_TOKEN", ""),
        help="GitHub token used for releases API requests.",
    )
    parser.add_argument(
        "--output",
        default="site/payload-index.json",
        help="Output path for payload-index.json.",
    )
    parser.add_argument(
        "--current-url",
        default=DEFAULT_CURRENT_INDEX_URL,
        help="Current published payload-index.json URL used in preserve-current mode.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    current_payload_index: JsonObject | None = None
    if args.mode == "preserve-current":
        current_payload_index = fetch_current_payload_index(args.current_url)
        if current_payload_index is not None:
            print(f"[payload-index] preserving published copy from {args.current_url}")
        else:
            print(
                "[payload-index] published copy unavailable; "
                "falling back to releases API",
            )

    releases = fetch_releases(args.repo, token=args.token)
    payload_index = choose_payload_index(
        args.mode,
        current_payload_index=current_payload_index,
        releases=releases,
        repo=args.repo,
    )
    write_payload_index(Path(args.output), payload_index)

    versions = [
        str(entry.get("version", ""))
        for entry in payload_index.get("payloads", [])
        if isinstance(entry, Mapping)
    ]
    newest = versions[0] if versions else "none"
    print(
        f"[payload-index] wrote {len(versions)} payloads to {args.output} "
        f"(newest: {newest})",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
