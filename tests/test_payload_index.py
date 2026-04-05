"""Tests for GitHub Pages payload-index generation helpers."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = _ROOT / "scripts"


@pytest.fixture
def payload_index_mod():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "payload_index",
        _SCRIPTS / "payload_index.py",
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_build_payload_index_from_releases_filters_and_sorts_versions(
    payload_index_mod,
) -> None:
    generated_at = dt.datetime(2026, 4, 6, 0, 0, 0, tzinfo=dt.UTC)
    releases = [
        {
            "tag_name": "v0.4.7",
            "draft": False,
            "prerelease": False,
            "assets": [{"name": "manifest.json"}],
        },
        {
            "tag_name": "v0.4.8",
            "draft": False,
            "prerelease": False,
            "assets": [{"name": "manifest.json"}],
        },
        {
            "tag_name": "v0.4.9",
            "draft": True,
            "prerelease": False,
            "assets": [{"name": "manifest.json"}],
        },
        {
            "tag_name": "v0.4.6",
            "draft": False,
            "prerelease": False,
            "assets": [{"name": "payload-aarch64-apple-darwin.zip"}],
        },
        {
            "tag_name": "launcher-v0.4.1",
            "draft": False,
            "prerelease": False,
            "assets": [{"name": "manifest.json"}],
        },
    ]

    payload_index = payload_index_mod.build_payload_index_from_releases(
        releases,
        "fruitiecutiepie/taskclf",
        generated_at=generated_at,
    )

    assert payload_index == {
        "kind": "payload-index",
        "schema_version": 1,
        "generated_at": "2026-04-06T00:00:00Z",
        "payloads": [
            {
                "version": "0.4.8",
                "manifest_url": "https://github.com/fruitiecutiepie/taskclf/releases/download/v0.4.8/manifest.json",
            },
            {
                "version": "0.4.7",
                "manifest_url": "https://github.com/fruitiecutiepie/taskclf/releases/download/v0.4.7/manifest.json",
            },
        ],
    }


def test_choose_payload_index_preserve_current_keeps_published_copy(
    payload_index_mod,
) -> None:
    current_payload_index = {
        "kind": "payload-index",
        "schema_version": 1,
        "generated_at": "2026-04-05T10:43:31Z",
        "payloads": [
            {
                "version": "0.4.7",
                "manifest_url": "https://github.com/fruitiecutiepie/taskclf/releases/download/v0.4.7/manifest.json",
            }
        ],
    }
    releases = [
        {
            "tag_name": "v0.4.8",
            "draft": False,
            "prerelease": False,
            "assets": [{"name": "manifest.json"}],
        }
    ]

    payload_index = payload_index_mod.choose_payload_index(
        "preserve-current",
        current_payload_index=current_payload_index,
        releases=releases,
        repo="fruitiecutiepie/taskclf",
    )

    assert payload_index == current_payload_index


def test_choose_payload_index_preserve_current_falls_back_when_invalid(
    payload_index_mod,
) -> None:
    releases = [
        {
            "tag_name": "v0.4.8",
            "draft": False,
            "prerelease": False,
            "assets": [{"name": "manifest.json"}],
        }
    ]

    payload_index = payload_index_mod.choose_payload_index(
        "preserve-current",
        current_payload_index={"kind": "not-payload-index"},
        releases=releases,
        repo="fruitiecutiepie/taskclf",
        generated_at=dt.datetime(2026, 4, 6, 1, 2, 3, tzinfo=dt.UTC),
    )

    assert payload_index["generated_at"] == "2026-04-06T01:02:03Z"
    assert payload_index["payloads"] == [
        {
            "version": "0.4.8",
            "manifest_url": "https://github.com/fruitiecutiepie/taskclf/releases/download/v0.4.8/manifest.json",
        }
    ]
