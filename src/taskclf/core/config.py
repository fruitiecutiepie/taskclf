"""User-level configuration persistence.

Stores per-install settings as a JSON file inside the data directory.
The file is created on first access with an auto-generated ``user_id``
(UUID) that never changes.

Typical location::

    data/processed/config.json

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
import uuid
from pathlib import Path
from typing import Any

from taskclf.core.defaults import DEFAULT_DATA_DIR

logger = logging.getLogger(__name__)

_DEFAULT_USERNAME = "default-user"
_CONFIG_FILENAME = "config.json"


class UserConfig:
    """Read/write access to ``config.json`` in a data directory.

    On first instantiation (no config file yet) a random UUID
    ``user_id`` is generated and persisted.  This ID is stable across
    restarts and is the value written into every ``LabelSpan``.

    ``username`` is a free-form display name that can be changed at any
    time without affecting label continuity.

    All mutations are persisted immediately.  The file is plain JSON so
    it can be hand-edited when the UI or CLI are not available.
    """

    def __init__(self, data_dir: Path | str = DEFAULT_DATA_DIR) -> None:
        self._path = Path(data_dir) / _CONFIG_FILENAME
        self._data: dict[str, Any] = self._load()
        self._ensure_user_id()

    def _load(self) -> dict[str, Any]:
        if self._path.exists():
            try:
                return json.loads(self._path.read_text("utf-8"))
            except (json.JSONDecodeError, OSError):
                logger.warning("Corrupt config at %s — using defaults", self._path)
        return {}

    def _ensure_user_id(self) -> None:
        """Generate and persist a stable ``user_id`` if one does not exist."""
        if "user_id" not in self._data:
            self._data["user_id"] = str(uuid.uuid4())
            self._persist()

    def _persist(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(self._data, indent=2) + "\n", "utf-8")

    # -- user_id (stable, read-only after creation) ----------------------------

    @property
    def user_id(self) -> str:
        """Stable UUID assigned to this install.  Never changes."""
        return self._data["user_id"]

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
