# User `config.toml` template

taskclf stores editable tray and UI settings in a TOML file next to your processed data directory (see [User Identity](../api/core/config.md)).

## First run

On first run, if `config.toml` does **not** exist, taskclf writes a **full commented starter file** once with every supported key and its default value. The file is **not** regenerated on later startups, so hand-edits and your chosen values are preserved.

Tray startup follows the same rule: creating the tray with an empty data directory produces that starter file.

Keys in the starter file follow a fixed order: **identity** (username), **notifications**, **ActivityWatch** (host, poll interval, HTTP timeout), **title hashing**, **web UI** (port, suggestion banner TTL), then **transition and gap-fill** thresholds. The same order is used when taskclf rewrites known keys.

## Review the canonical example

- **Published guide (this site):** you are reading it; the authoritative key list is in [User Identity](../api/core/config.md).
- **Repository template file (raw):** [user_config.template.toml](https://github.com/fruitiecutiepie/taskclf/blob/master/configs/user_config.template.toml)
- **GitHub Pages:** after docs deploy, this page is available at [https://fruitiecutiepie.github.io/taskclf/guide/config_template/](https://fruitiecutiepie.github.io/taskclf/guide/config_template/)

## What still updates the file

- Editing settings through the web UI where the API persists them (for example username or suggestion banner TTL) merges into `config.toml`.
- Setting `--username` on `taskclf tray` persists the display name when provided.
- Any direct edit in a text editor.

The app does **not** rewrite the whole file on every launch to match resolved CLI/runtime defaults.

## Relation to pipeline configs

The files under `configs/` in the repository (for example `config.yml` for training features) are **not** the same as per-user `config.toml`. See [configs/README](https://github.com/fruitiecutiepie/taskclf/blob/master/configs/README.md).
