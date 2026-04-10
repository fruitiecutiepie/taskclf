"""Tests for taskclf.core.config.UserConfig."""

from __future__ import annotations

import json
import tomllib
import uuid

import pytest

from pathlib import Path

from taskclf.core.config import (
    UserConfig,
    render_default_user_config_toml,
    default_starter_config_dict,
)


def test_auto_generates_uuid_user_id(tmp_path):
    cfg = UserConfig(tmp_path)
    uid = cfg.user_id
    uuid.UUID(uid)
    assert len(uid) == 36


def test_user_id_stored_in_dot_file(tmp_path):
    """user_id lives in .user_id, not in config.toml."""
    cfg = UserConfig(tmp_path)
    assert (tmp_path / ".user_id").exists()
    assert (tmp_path / ".user_id").read_text().strip() == cfg.user_id
    assert (tmp_path / "config.toml").exists()
    toml_text = (tmp_path / "config.toml").read_text()
    assert "user_id" not in toml_text


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
    (tmp_path / "config.toml").write_text("NOT VALID TOML ][", "utf-8")
    cfg = UserConfig(tmp_path)
    assert cfg.username == "default-user"
    uuid.UUID(cfg.user_id)


def test_config_creates_parent_dirs(tmp_path):
    deep = tmp_path / "a" / "b" / "c"
    cfg = UserConfig(deep)
    cfg.username = "nested"
    assert (deep / "config.toml").exists()
    data = tomllib.loads((deep / "config.toml").read_text())
    assert data["username"] == "nested"
    assert "user_id" not in data
    assert (deep / ".user_id").exists()
    uuid.UUID(cfg.user_id)


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


def test_empty_user_id_file_regenerated(tmp_path):
    """Regression: empty .user_id file must trigger regeneration."""
    (tmp_path / ".user_id").write_text("", "utf-8")
    cfg = UserConfig(tmp_path)
    assert cfg.user_id != ""
    uuid.UUID(cfg.user_id)


def test_missing_user_id_file_regenerated(tmp_path):
    """Regression: missing .user_id file must trigger generation."""
    (tmp_path / "config.toml").write_text('username = "alice"\n')
    cfg = UserConfig(tmp_path)
    assert cfg.user_id is not None
    assert cfg.user_id != ""
    uuid.UUID(cfg.user_id)


def test_user_id_migrated_from_toml_to_dot_file(tmp_path):
    """If old code wrote user_id into config.toml, move it to .user_id."""
    (tmp_path / "config.toml").write_text(
        'user_id = "in-toml-uuid"\nusername = "bob"\n'
    )
    cfg = UserConfig(tmp_path)
    assert cfg.user_id == "in-toml-uuid"
    assert (tmp_path / ".user_id").read_text().strip() == "in-toml-uuid"
    toml_text = (tmp_path / "config.toml").read_text()
    assert "user_id" not in toml_text


# ---------------------------------------------------------------------------
# JSON -> TOML migration
# ---------------------------------------------------------------------------


def test_json_to_toml_migration(tmp_path):
    """Existing config.json is migrated to config.toml + .user_id."""
    json_path = tmp_path / "config.json"
    json_path.write_text(
        json.dumps(
            {
                "user_id": "migrated-uuid",
                "username": "migrator",
                "poll_seconds": 90,
            }
        )
    )

    cfg = UserConfig(tmp_path)

    assert cfg.user_id == "migrated-uuid"
    assert cfg.username == "migrator"
    assert cfg.as_dict()["poll_seconds"] == 90
    assert (tmp_path / "config.toml").exists()
    assert (tmp_path / ".user_id").exists()
    assert not (tmp_path / "config.json").exists()
    assert (tmp_path / "config.json.bak").exists()
    toml_text = (tmp_path / "config.toml").read_text()
    assert "user_id" not in toml_text


def test_json_migration_strips_help_key(tmp_path):
    """The _help dict from JSON-era configs is dropped during migration."""
    json_path = tmp_path / "config.json"
    json_path.write_text(
        json.dumps(
            {
                "user_id": "help-uuid",
                "_help": {"user_id": "old help text"},
                "poll_seconds": 60,
            }
        )
    )

    cfg = UserConfig(tmp_path)
    assert "_help" not in cfg.as_dict()
    toml_text = (tmp_path / "config.toml").read_text()
    assert "_help" not in toml_text


def test_json_not_migrated_when_toml_exists(tmp_path):
    """If config.toml already exists, config.json is left alone."""
    (tmp_path / ".user_id").write_text("toml-uuid", "utf-8")
    (tmp_path / "config.toml").write_text('username = "from-toml"\n')
    (tmp_path / "config.json").write_text(json.dumps({"user_id": "json-uuid"}))

    cfg = UserConfig(tmp_path)
    assert cfg.user_id == "toml-uuid"
    assert (tmp_path / "config.json").exists()


def test_toml_has_comments(tmp_path):
    """Persisted config.toml contains # comment lines for known settings."""
    cfg = UserConfig(tmp_path)
    cfg.update({"notifications_enabled": True, "poll_seconds": 60})
    text = (tmp_path / "config.toml").read_text()
    assert "# --- Notifications ---" in text
    assert "# If false, suppresses tray desktop" in text
    assert "# --- ActivityWatch ---" in text
    assert "user_id" not in text


def test_starter_template_written_on_first_run(tmp_path):
    """First UserConfig creates a full starter config with every known key."""
    UserConfig(tmp_path)
    data = tomllib.loads((tmp_path / "config.toml").read_text())
    assert data == default_starter_config_dict()


def test_title_secret_generated_and_persisted(tmp_path):
    """Fresh installs create a local .title_secret that stays out of config.toml."""
    cfg = UserConfig(tmp_path)

    secret_path = tmp_path / ".title_secret"
    assert secret_path.exists()
    assert secret_path.read_text("utf-8").strip() == cfg.title_secret
    assert cfg.title_salt == cfg.title_secret

    config_text = (tmp_path / "config.toml").read_text("utf-8")
    assert "title_salt" not in config_text
    assert cfg.title_secret not in config_text


def test_legacy_title_salt_migrated_to_title_secret(tmp_path):
    """Legacy config.toml title_salt is moved into .title_secret and removed from TOML."""
    (tmp_path / "config.toml").write_text('title_salt = "legacy-salt"\n', "utf-8")

    cfg = UserConfig(tmp_path)

    assert cfg.title_secret == "legacy-salt"
    assert (tmp_path / ".title_secret").read_text("utf-8").strip() == "legacy-salt"
    assert "title_salt" not in (tmp_path / "config.toml").read_text("utf-8")


def test_title_secret_not_exposed_in_as_dict(tmp_path):
    """Runtime config payloads exclude both title_secret and legacy title_salt."""
    cfg = UserConfig(tmp_path)
    data = cfg.as_dict()
    assert "title_secret" not in data
    assert "title_salt" not in data


def test_existing_config_not_regenerated(tmp_path):
    """An existing config.toml is not overwritten on a second UserConfig load."""
    path = tmp_path / "config.toml"
    path.write_text('username = "keep-me"\npoll_seconds = 77\n', "utf-8")
    (tmp_path / ".user_id").write_text("fixed-uuid", "utf-8")

    before = path.read_text()
    UserConfig(tmp_path)
    assert path.read_text() == before


def test_template_file_matches_render():
    """configs/user_config.template.toml stays in sync with render_default_user_config_toml."""
    root = Path(__file__).resolve().parents[1]
    on_disk = (root / "configs" / "user_config.template.toml").read_text()
    assert on_disk == render_default_user_config_toml()
