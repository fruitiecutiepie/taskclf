"""User-level configuration persistence.

Stores editable settings in ``config.toml`` and the immutable install
identity (``user_id``) in a separate ``.user_id`` file so that users
cannot accidentally break label continuity by editing their config.

Typical locations::

    data/processed/config.toml   # user-editable settings
    data/processed/.user_id      # stable UUID, never shown in config

Usage::

    from taskclf.core.config import UserConfig

    cfg = UserConfig(data_dir)
    cfg.user_id    # stable UUID, auto-generated on first run
    cfg.username   # editable display name (defaults to "default-user")
    cfg.username = "alice"   # persists immediately
    cfg.as_dict()
"""

from __future__ import annotations

import json
import logging
import tomllib
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import tomli_w

from taskclf.core.defaults import DEFAULT_DATA_DIR

logger = logging.getLogger(__name__)

_DEFAULT_USERNAME = "default-user"
_CONFIG_FILENAME = "config.toml"
_USER_ID_FILENAME = ".user_id"
_LEGACY_JSON_FILENAME = "config.json"

_SETTING_COMMENTS: dict[str, str] = {
    "username": "Display name shown in labels. Does not affect label identity.",
    "notifications_enabled": "Set to false to suppress all desktop notifications.",
    "privacy_notifications": "When true, app names are redacted from notifications.",
    "poll_seconds": "Seconds between ActivityWatch polling cycles.",
    "transition_minutes": "Minutes a new app must persist before a transition fires.",
    "aw_host": "ActivityWatch server URL.",
    "title_salt": "Salt used for hashing window titles (privacy).",
    "ui_port": "Port for the embedded web UI server.",
}


@dataclass(eq=False)
class UserConfig:
    """Read/write access to ``config.toml`` and ``.user_id`` in a data directory.

    The immutable ``user_id`` lives in a separate ``.user_id`` file so
    it never appears in the user-editable config.  On first run a random
    UUID is generated and written there.

    ``username`` is a free-form display name that can be changed at any
    time without affecting label continuity.

    All mutations are persisted immediately.  ``config.toml`` uses ``#``
    comments above each setting so it can be hand-edited.
    """

    data_dir: Path | str = DEFAULT_DATA_DIR
    _dir: Path = field(init=False)
    _path: Path = field(init=False)
    _user_id_path: Path = field(init=False)
    _data: dict[str, Any] = field(init=False)
    _uid: str = field(init=False)

    def __post_init__(self) -> None:
        self._dir = Path(self.data_dir)
        self._path = self._dir / _CONFIG_FILENAME
        self._user_id_path = self._dir / _USER_ID_FILENAME
        self._migrate_json()
        self._data = self._load()
        self._uid = self._load_user_id()

    # -- migration & loading ---------------------------------------------------

    def _migrate_json(self) -> None:
        """One-time migration from config.json to config.toml + .user_id."""
        json_path = self._dir / _LEGACY_JSON_FILENAME
        if self._path.exists() or not json_path.exists():
            return
        try:
            data = json.loads(json_path.read_text("utf-8"))
            data.pop("_help", None)

            uid = data.pop("user_id", None)
            if uid:
                self._dir.mkdir(parents=True, exist_ok=True)
                self._user_id_path.write_text(uid, "utf-8")

            self._dir.mkdir(parents=True, exist_ok=True)
            self._path.write_text(_to_commented_toml(data), "utf-8")
            json_path.rename(json_path.with_suffix(".json.bak"))
            logger.info(
                "Migrated %s → %s + %s", json_path, self._path, self._user_id_path
            )
        except Exception:
            logger.warning("Failed to migrate %s", json_path, exc_info=True)

    def _load(self) -> dict[str, Any]:
        if self._path.exists():
            try:
                return dict(tomllib.loads(self._path.read_text("utf-8")))
            except tomllib.TOMLDecodeError, OSError:
                logger.warning("Corrupt config at %s — using defaults", self._path)
        return {}

    def _load_user_id(self) -> str:
        """Read or generate the stable user_id from .user_id file."""
        if self._user_id_path.exists():
            uid = self._user_id_path.read_text("utf-8").strip()
            if uid:
                return uid

        # Migrate from config.toml if it was written there by old code
        uid = self._data.pop("user_id", None)
        if uid:
            self._persist_user_id(uid)
            self._persist()
            return uid

        uid = str(uuid.uuid4())
        self._persist_user_id(uid)
        return uid

    def _persist_user_id(self, uid: str) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        self._user_id_path.write_text(uid, "utf-8")

    def _persist(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(_to_commented_toml(self._data), "utf-8")

    # -- user_id (stable, read-only after creation) ----------------------------

    @property
    def user_id(self) -> str:
        """Stable UUID assigned to this install.  Never changes."""
        return self._uid

    # -- username (editable display name) --------------------------------------

    @property
    def username(self) -> str:
        return self._data.get("username", _DEFAULT_USERNAME)

    @username.setter
    def username(self, value: str) -> None:
        value = value.strip()
        if not value:
            raise ValueError("username must not be empty")
        self._data["username"] = value
        self._persist()

    # -- generic helpers -------------------------------------------------------

    def as_dict(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "username": self.username,
            **{k: v for k, v in self._data.items() if k not in ("user_id", "username")},
        }

    def update(self, patch: dict[str, Any]) -> dict[str, Any]:
        """Merge *patch* into the config and persist.  Returns the full config.

        ``user_id`` is ignored in *patch* — it is immutable after creation.
        """
        if "username" in patch:
            name = str(patch["username"]).strip()
            if not name:
                raise ValueError("username must not be empty")
            self._data["username"] = name
        for key, val in patch.items():
            if key in ("user_id", "username"):
                continue
            self._data[key] = val
        self._persist()
        return self.as_dict()


def _to_commented_toml(data: dict[str, Any]) -> str:
    """Serialize *data* as TOML with ``#`` comments above known settings."""
    lines: list[str] = []
    for key, val in data.items():
        if key == "user_id":
            continue
        comment = _SETTING_COMMENTS.get(key)
        if comment:
            lines.append(f"# {comment}")
        lines.append(tomli_w.dumps({key: val}).rstrip())
        lines.append("")
    return "\n".join(lines) + "\n" if lines else ""
