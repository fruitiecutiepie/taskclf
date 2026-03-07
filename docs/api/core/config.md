# core.config

User-level configuration persistence.

Stores editable settings in a TOML file and the immutable install
identity (`user_id`) in a separate `.user_id` file so that users
cannot accidentally break label continuity by editing their config.

On first run a unique UUID `user_id` is generated and written to
`.user_id`.  This stable ID is written into every `LabelSpan` and
never changes.

The editable `username` field is a display name that can be changed
freely without affecting label identity or continuity.

## Location

```
<data_dir>/config.toml   # user-editable settings
<data_dir>/.user_id      # stable UUID (auto-generated, do not edit)
```

Default: `~/Library/Application Support/taskclf/data/processed/`

## config.toml schema

```toml
# Display name shown in labels. Does not affect label identity.
username = "alice"

# Set to false to suppress all desktop notifications.
notifications_enabled = true

# When true, app names are redacted from notifications.
privacy_notifications = true

# Seconds between ActivityWatch polling cycles.
poll_seconds = 60

# Minutes a new app must persist before a transition fires.
transition_minutes = 3

# ActivityWatch server URL.
aw_host = "http://localhost:5600"

# Salt used for hashing window titles (privacy).
title_salt = "taskclf-default-salt"

# Port for the embedded web UI server.
ui_port = 8741
```

| Key | Type | Default | Description |
|---|---|---|---|
| `username` | `str` | `"default-user"` | Display name (cosmetic only) |
| `notifications_enabled` | `bool` | `true` | Desktop notifications on/off |
| `privacy_notifications` | `bool` | `true` | Redact app names from notifications |
| `poll_seconds` | `int` | `60` | AW polling interval |
| `transition_minutes` | `int` | `3` | App persistence threshold for transitions |
| `aw_host` | `str` | `"http://localhost:5600"` | ActivityWatch server URL |
| `title_salt` | `str` | `"taskclf-default-salt"` | Window title hash salt |
| `ui_port` | `int` | `8741` | Web UI server port |

The `user_id` UUID is stored separately in `.user_id` and never appears in `config.toml`.

## Backward compatibility

On startup, if `config.json` exists but `config.toml` does not, the JSON
file is automatically migrated: settings go to `config.toml`, `user_id`
goes to `.user_id`, and the original is renamed to `config.json.bak`.

If old code wrote `user_id` into `config.toml`, it is automatically
moved to `.user_id` and removed from the TOML file.

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
cfg.user_id            # stable UUID (read-only, from .user_id)
cfg.username           # "default-user"
cfg.username = "alice"  # persists immediately to config.toml
cfg.as_dict()          # {"user_id": "...", "username": "alice", ...}
```

::: taskclf.core.config
