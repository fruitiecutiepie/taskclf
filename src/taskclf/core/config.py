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

from taskclf.core.defaults import (
    DEFAULT_AW_HOST,
    DEFAULT_AW_TIMEOUT_SECONDS,
    DEFAULT_DATA_DIR,
    DEFAULT_IDLE_TRANSITION_MINUTES,
    DEFAULT_POLL_SECONDS,
    DEFAULT_TITLE_SALT,
    DEFAULT_TRANSITION_MINUTES,
)

logger = logging.getLogger(__name__)

_DEFAULT_USERNAME = "default-user"
_CONFIG_FILENAME = "config.toml"
_USER_ID_FILENAME = ".user_id"
_LEGACY_JSON_FILENAME = "config.json"

# Default UI port (tray / embedded server); kept here so starter template matches runtime.
_DEFAULT_UI_PORT: int = 8741
_DEFAULT_GAP_FILL_ESCALATION_MINUTES: int = 480


@dataclass(frozen=True, slots=True)
class UserConfigField:
    """One user-editable key in ``config.toml`` with default and comment."""

    key: str
    default: Any
    comment: str
    section_title: str | None = None


# Ordered full starter template: identity → notifications → ActivityWatch → privacy →
# web UI → transition/gap behavior. Same order is used when persisting known keys.
# Section titles appear as ``# --- Title ---`` on the first key of each group.
USER_CONFIG_FIELDS: tuple[UserConfigField, ...] = (
    UserConfigField(
        "username",
        _DEFAULT_USERNAME,
        "Display name in exported labels; cosmetic only (stable identity is in a separate file).",
        section_title="Identity",
    ),
    UserConfigField(
        "notifications_enabled",
        True,
        "If false, suppresses tray desktop notifications.",
        section_title="Notifications",
    ),
    UserConfigField(
        "privacy_notifications",
        True,
        "If true, notification text hides raw app names (recommended for screen sharing).",
    ),
    UserConfigField(
        "aw_host",
        DEFAULT_AW_HOST,
        "Base URL of your ActivityWatch server (typically http://127.0.0.1:5600).",
        section_title="ActivityWatch",
    ),
    UserConfigField(
        "poll_seconds",
        DEFAULT_POLL_SECONDS,
        "How often the tray asks ActivityWatch for the active window (seconds).",
    ),
    UserConfigField(
        "aw_timeout_seconds",
        DEFAULT_AW_TIMEOUT_SECONDS,
        "HTTP timeout for ActivityWatch API calls (seconds).",
    ),
    UserConfigField(
        "title_salt",
        DEFAULT_TITLE_SALT,
        "Salt for hashing window titles before features; change if you rotate privacy.",
        section_title="Privacy",
    ),
    UserConfigField(
        "ui_port",
        _DEFAULT_UI_PORT,
        "TCP port for the embedded labeling dashboard (http://127.0.0.1:this port).",
        section_title="Web UI",
    ),
    UserConfigField(
        "suggestion_banner_ttl_seconds",
        0,
        "Auto-dismiss the model suggestion banner after N seconds; 0 keeps it until you act.",
    ),
    UserConfigField(
        "transition_minutes",
        DEFAULT_TRANSITION_MINUTES,
        "How long a foreground app must stay dominant before a transition is detected.",
        section_title="Transitions and gaps",
    ),
    UserConfigField(
        "idle_transition_minutes",
        DEFAULT_IDLE_TRANSITION_MINUTES,
        "Shorter threshold for lockscreen/idle apps (BreakIdle); often below transition_minutes.",
    ),
    UserConfigField(
        "gap_fill_escalation_minutes",
        _DEFAULT_GAP_FILL_ESCALATION_MINUTES,
        "Unlabeled minutes before the tray shows gap-fill escalation (orange icon).",
    ),
)

_DEFAULT_STARTER_DICT: dict[str, Any] = {f.key: f.default for f in USER_CONFIG_FIELDS}

# Prepended to generated config.toml and the repo template; points to canonical remote copies.
_USER_CONFIG_TEMPLATE_REMOTE_HEADER = (
    "# Canonical template:\n"
    "#   GitHub: https://github.com/fruitiecutiepie/taskclf/blob/master/configs/user_config.template.toml\n"
    "#   Download: https://raw.githubusercontent.com/fruitiecutiepie/taskclf/master/configs/user_config.template.toml\n"
    "#   Guide: https://fruitiecutiepie.github.io/taskclf/guide/config_template/\n"
    "\n"
)


def default_starter_config_dict() -> dict[str, Any]:
    """Return a copy of the full default ``config.toml`` key set and values."""
    return dict(_DEFAULT_STARTER_DICT)


def render_default_user_config_toml() -> str:
    """Return the full commented default ``config.toml`` text (for docs and templates)."""
    return _USER_CONFIG_TEMPLATE_REMOTE_HEADER + _to_commented_toml(
        _DEFAULT_STARTER_DICT
    )


def _comment_for_key(key: str) -> str | None:
    for f in USER_CONFIG_FIELDS:
        if f.key == key:
            return f.comment
    return None


def _user_config_field_spec(key: str) -> UserConfigField | None:
    for f in USER_CONFIG_FIELDS:
        if f.key == key:
            return f
    return None


def _ordered_keys_for_persist(data: dict[str, Any]) -> list[str]:
    """Emit known keys in spec order, then any remaining keys (e.g. custom) sorted."""
    spec_order = [f.key for f in USER_CONFIG_FIELDS]
    ordered: list[str] = []
    seen: set[str] = set()
    for k in spec_order:
        if k in data and k != "user_id":
            ordered.append(k)
            seen.add(k)
    for k in sorted(data.keys()):
        if k in ("user_id",) or k in seen:
            continue
        ordered.append(k)
    return ordered


@dataclass(eq=False)
class UserConfig:
    """Read/write access to ``config.toml`` and ``.user_id`` in a data directory.

    The immutable ``user_id`` lives in a separate ``.user_id`` file so
    it never appears in the user-editable config.  On first run a random
    UUID is generated and written there.

    ``username`` is a free-form display name that can be changed at any
    time without affecting label continuity.

    On first run, if ``config.toml`` is missing, a full commented starter
    template is written once. Existing files are not regenerated on later
    loads.

    Mutations from ``update()`` or property setters persist immediately.
    ``config.toml`` uses ``#`` comments above each known setting.
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
        self._ensure_starter_config_if_missing()

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

    def _ensure_starter_config_if_missing(self) -> None:
        """Write the full default template only when ``config.toml`` does not exist."""
        if self._path.exists():
            return
        self._dir.mkdir(parents=True, exist_ok=True)
        self._path.write_text(render_default_user_config_toml(), "utf-8")
        self._data = self._load()

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
    """Serialize *data* as TOML with ``#`` section headers and line comments for known keys."""
    lines: list[str] = []
    for key in _ordered_keys_for_persist(data):
        if key == "user_id":
            continue
        val = data[key]
        spec = _user_config_field_spec(key)
        if spec and spec.section_title:
            if lines:
                lines.append("")
            lines.append(f"# --- {spec.section_title} ---")
        comment = _comment_for_key(key)
        if comment:
            lines.append(f"# {comment}")
        lines.append(tomli_w.dumps({key: val}).rstrip())
        lines.append("")
    return "\n".join(lines) + "\n" if lines else ""
