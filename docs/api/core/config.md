# core.config

User-level configuration persistence.

Stores per-install settings as a JSON file inside the data directory.
On first run a unique UUID `user_id` is generated and persisted.  This
stable ID is written into every `LabelSpan` and never changes.

The editable `username` field is a display name that can be changed
freely without affecting label identity or continuity.

## Location

```
<data_dir>/config.json
```

Default: `data/processed/config.json`

## Schema

```json
{
  "user_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "username": "alice"
}
```

| Key | Type | Default | Mutable | Description |
|---|---|---|---|---|
| `user_id` | `str` | auto-generated UUID | No | Stable identity written into labels |
| `username` | `str` | `"default-user"` | Yes | Display name (cosmetic only) |

## CLI usage

Set display name at tray launch (persisted for future runs):

```bash
taskclf tray --username alice
```

## REST API

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/config/user` | Read user_id + username |
| `PUT` | `/api/config/user` | Update username |

### `GET /api/config/user`

Returns:

```json
{
  "user_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "username": "alice"
}
```

### `PUT /api/config/user`

Body (`user_id` is ignored -- it is immutable):

```json
{"username": "bob"}
```

Returns the full updated config.

## Python usage

```python
from taskclf.core.config import UserConfig

cfg = UserConfig("data/processed")
cfg.user_id            # stable UUID (read-only)
cfg.username           # "default-user"
cfg.username = "alice"  # persists immediately
cfg.as_dict()          # {"user_id": "...", "username": "alice"}
```

::: taskclf.core.config
