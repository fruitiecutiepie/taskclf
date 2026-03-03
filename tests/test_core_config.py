"""Tests for taskclf.core.config.UserConfig."""

from __future__ import annotations

import json
import uuid

import pytest

from taskclf.core.config import UserConfig


def test_auto_generates_uuid_user_id(tmp_path):
    cfg = UserConfig(tmp_path)
    uid = cfg.user_id
    uuid.UUID(uid)
    assert len(uid) == 36


def test_user_id_stable_across_reloads(tmp_path):
    uid1 = UserConfig(tmp_path).user_id
    uid2 = UserConfig(tmp_path).user_id
    assert uid1 == uid2


def test_user_id_is_immutable_via_update(tmp_path):
    cfg = UserConfig(tmp_path)
    original = cfg.user_id
    cfg.update({"user_id": "should-be-ignored"})
    assert cfg.user_id == original


def test_default_username(tmp_path):
    cfg = UserConfig(tmp_path)
    assert cfg.username == "default-user"


def test_set_username_persists(tmp_path):
    cfg = UserConfig(tmp_path)
    cfg.username = "alice"
    assert cfg.username == "alice"

    reloaded = UserConfig(tmp_path)
    assert reloaded.username == "alice"


def test_empty_username_rejected(tmp_path):
    cfg = UserConfig(tmp_path)
    with pytest.raises(ValueError, match="empty"):
        cfg.username = ""
    with pytest.raises(ValueError, match="empty"):
        cfg.username = "   "


def test_as_dict_contains_both(tmp_path):
    cfg = UserConfig(tmp_path)
    d = cfg.as_dict()
    assert "user_id" in d
    assert d["username"] == "default-user"

    cfg.username = "bob"
    assert cfg.as_dict()["username"] == "bob"
    assert cfg.as_dict()["user_id"] == cfg.user_id


def test_update_username(tmp_path):
    cfg = UserConfig(tmp_path)
    result = cfg.update({"username": "carol"})
    assert result["username"] == "carol"
    assert cfg.username == "carol"


def test_update_empty_username_rejected(tmp_path):
    cfg = UserConfig(tmp_path)
    with pytest.raises(ValueError, match="empty"):
        cfg.update({"username": ""})


def test_update_no_op(tmp_path):
    cfg = UserConfig(tmp_path)
    cfg.username = "dave"
    result = cfg.update({})
    assert result["username"] == "dave"


def test_corrupt_config_falls_back(tmp_path):
    (tmp_path / "config.json").write_text("NOT VALID JSON", "utf-8")
    cfg = UserConfig(tmp_path)
    assert cfg.username == "default-user"
    uuid.UUID(cfg.user_id)


def test_config_creates_parent_dirs(tmp_path):
    deep = tmp_path / "a" / "b" / "c"
    cfg = UserConfig(deep)
    cfg.username = "nested"
    assert (deep / "config.json").exists()
    data = json.loads((deep / "config.json").read_text())
    assert data["username"] == "nested"
    uuid.UUID(data["user_id"])


def test_changing_username_does_not_change_user_id(tmp_path):
    cfg = UserConfig(tmp_path)
    original_id = cfg.user_id
    cfg.username = "one"
    cfg.username = "two"
    cfg.username = "three"
    assert cfg.user_id == original_id


# ---------------------------------------------------------------------------
# Edge cases: custom keys and as_dict
# ---------------------------------------------------------------------------


def test_update_custom_keys_persisted(tmp_path):
    """TC-CFG-001: update() with custom keys stores and persists them."""
    cfg = UserConfig(tmp_path)
    result = cfg.update({"theme": "dark", "poll_interval": 30})
    assert result["theme"] == "dark"
    assert result["poll_interval"] == 30

    reloaded = UserConfig(tmp_path)
    assert reloaded.as_dict()["theme"] == "dark"
    assert reloaded.as_dict()["poll_interval"] == 30


def test_as_dict_includes_custom_keys(tmp_path):
    """TC-CFG-002: as_dict() surfaces extra keys alongside user_id and username."""
    cfg = UserConfig(tmp_path)
    cfg.update({"locale": "en_US"})
    d = cfg.as_dict()
    assert "user_id" in d
    assert "username" in d
    assert d["locale"] == "en_US"


def test_update_whitespace_username_rejected(tmp_path):
    """TC-CFG-003: update() with whitespace-only username raises ValueError."""
    cfg = UserConfig(tmp_path)
    with pytest.raises(ValueError, match="empty"):
        cfg.update({"username": "   "})


def test_empty_user_id_regenerated(tmp_path):
    """Regression: empty-string user_id in config.json must be regenerated."""
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"user_id": ""}))
    cfg = UserConfig(tmp_path)
    assert cfg.user_id != ""
    uuid.UUID(cfg.user_id)


def test_null_user_id_regenerated(tmp_path):
    """Regression: null user_id in config.json must be regenerated."""
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"user_id": None}))
    cfg = UserConfig(tmp_path)
    assert cfg.user_id is not None
    assert cfg.user_id != ""
    uuid.UUID(cfg.user_id)
