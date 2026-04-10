# core.config

User-level configuration persistence.

Stores editable settings in a TOML file and the immutable install
identity (`user_id`) in a separate `.user_id` file so that users
cannot accidentally break label continuity by editing their config.

On first run a unique UUID `user_id` is generated and written to
`.user_id`.  This stable ID is written into every `LabelSpan` and
never changes.
`UserConfig` is a dataclass (`UserConfig(data_dir=...)`).

The editable `username` field is a display name that can be changed
freely without affecting label identity or continuity.

## Location

```
<data_dir>/config.toml   # user-editable settings
<data_dir>/.user_id      # stable UUID (auto-generated, do not edit)
<data_dir>/.title_secret # local-only secret for title hashing/sketching
```

Default: `~/Library/Application Support/taskclf/data/processed/`

## config.toml schema

On first run, if `config.toml` is missing, taskclf writes a **full commented starter file** once (see the [User config template](../../guide/config_template.md) guide and [`configs/user_config.template.toml`](https://github.com/fruitiecutiepie/taskclf/blob/master/configs/user_config.template.toml)). The file is not regenerated on later startups.

```toml
# Canonical template:
#   GitHub: https://github.com/fruitiecutiepie/taskclf/blob/master/configs/user_config.template.toml
#   Download: https://raw.githubusercontent.com/fruitiecutiepie/taskclf/master/configs/user_config.template.toml
#   Guide: https://fruitiecutiepie.github.io/taskclf/guide/config_template/

# --- Identity ---
# Display name in exported labels; cosmetic only (stable identity is in a separate file).
username = "default-user"


# --- Notifications ---
# If false, suppresses tray desktop notifications.
notifications_enabled = true

# If true, notification text hides raw app names (recommended for screen sharing).
privacy_notifications = true


# --- ActivityWatch ---
# Base URL of your ActivityWatch server (typically http://127.0.0.1:5600).
aw_host = "http://localhost:5600"

# How often the tray asks ActivityWatch for the active window (seconds).
poll_seconds = 60

# HTTP timeout for ActivityWatch API calls (seconds).
aw_timeout_seconds = 10


# --- Web UI ---
# TCP port for the embedded labeling dashboard (http://127.0.0.1:this port).
ui_port = 8741

# Auto-dismiss the model suggestion banner after N seconds; 0 keeps it until you act.
suggestion_banner_ttl_seconds = 0


# --- Transitions and gaps ---
# How long a foreground app must stay dominant before a transition is detected.
transition_minutes = 2

# Shorter threshold for lockscreen/idle apps (BreakIdle); often below transition_minutes.
idle_transition_minutes = 1

# Unlabeled minutes before the tray shows gap-fill escalation (orange icon).
gap_fill_escalation_minutes = 480
```

| Key | Type | Default | Description |
|---|---|---|---|
| `username` | `str` | `"default-user"` | Display name in labels (stable UUID is in `.user_id`) |
| `notifications_enabled` | `bool` | `true` | Tray desktop notifications on/off |
| `privacy_notifications` | `bool` | `true` | Hide raw app names in notification text |
| `aw_host` | `str` | `"http://localhost:5600"` | ActivityWatch server base URL |
| `poll_seconds` | `int` | `60` | Polling interval for the active window (seconds) |
| `aw_timeout_seconds` | `int` | `10` | ActivityWatch HTTP timeout (seconds) |
| `ui_port` | `int` | `8741` | Embedded labeling dashboard TCP port |
| `suggestion_banner_ttl_seconds` | `int` | `0` | Auto-dismiss suggestion banner after N seconds; `0` until user acts |
| `transition_minutes` | `int` | `2` | Dominance time before a transition is detected (minutes) |
| `idle_transition_minutes` | `int` | `1` | Threshold for lockscreen/idle / BreakIdle (minutes) |
| `gap_fill_escalation_minutes` | `int` | `480` | Unlabeled time before gap-fill escalation (minutes) |

The `user_id` UUID is stored separately in `.user_id` and never appears in `config.toml`.
The title-featurization secret is stored separately in `.title_secret` and is intentionally omitted from `config.toml`, REST payloads, and `UserConfig.as_dict()`.

## Backward compatibility

On startup, if `config.json` exists but `config.toml` does not, the JSON
file is automatically migrated: settings go to `config.toml`, `user_id`
goes to `.user_id`, and the original is renamed to `config.json.bak`.

If old code wrote `user_id` into `config.toml`, it is automatically
moved to `.user_id` and removed from the TOML file.

Legacy `title_salt` values are also migrated out of `config.toml` into
`.title_secret`.  `UserConfig.title_salt` remains as a read-only
compatibility alias for code paths that still expect that name.

## CLI usage

Set display name at tray launch (persisted for future runs):

```bash
taskclf tray --username alice
```

## REST API

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/config/user` | Read user_id, username, and `suggestion_banner_ttl_seconds` |
| `PUT` | `/api/config/user` | Update username and/or `suggestion_banner_ttl_seconds` |

### `GET /api/config/user`

Returns:

```json
{
  "user_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "username": "alice",
  "suggestion_banner_ttl_seconds": 0
}
```

### `PUT /api/config/user`

Body (`user_id` is ignored -- it is immutable):

```json
{"username": "bob", "suggestion_banner_ttl_seconds": 600}
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
