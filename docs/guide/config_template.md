# User `config.toml` template

taskclf stores editable tray and UI settings in a TOML file next to your processed data directory (see [User Identity](../api/core/config.md)).

## First run

On first run, if `config.toml` does **not** exist, taskclf writes a **full commented starter file** once with every supported key and its default value. The file is **not** regenerated on later startups, so hand-edits and your chosen values are preserved.

Tray startup follows the same rule: creating the tray with an empty data directory produces that starter file.

The starter file groups settings under **section headers** like `# --- Identity ---`, then one line of help text per key. The same order and headings are used when taskclf rewrites known keys.

## Review the canonical example

The same three **remote URLs** (labeled GitHub, Download, and Guide in the file) are copied into the top of a newly generated `config.toml` as comments, so you can open the live template from your local file.

The checked-in `configs/user_config.template.toml` file is kept in sync with the renderer that writes the first-run starter file.

| | URL |
|---|---|
| **Browse on GitHub** (default branch) | [github.com/fruitiecutiepie/taskclf/blob/master/configs/user_config.template.toml](https://github.com/fruitiecutiepie/taskclf/blob/master/configs/user_config.template.toml) |
| **Raw file** (for curl/wget or “Save link as…”) | [raw.githubusercontent.com/fruitiecutiepie/taskclf/master/configs/user_config.template.toml](https://raw.githubusercontent.com/fruitiecutiepie/taskclf/master/configs/user_config.template.toml) |
| **This guide on GitHub Pages** | [fruitiecutiepie.github.io/taskclf/guide/config_template/](https://fruitiecutiepie.github.io/taskclf/guide/config_template/) |

Authoritative key reference: [User Identity](../api/core/config.md).

## What still updates the file

- Editing settings through the web UI where the API persists them (for example username or suggestion banner TTL) merges into `config.toml`.
- Setting `--username` on `taskclf tray` persists the display name when provided.
- Any direct edit in a text editor.

The app does **not** rewrite the whole file on every launch to match resolved CLI/runtime defaults.

## Relation to pipeline configs

The files under `configs/` in the repository (for example `config.yml` for training features) are **not** the same as per-user `config.toml`. See [configs/README](https://github.com/fruitiecutiepie/taskclf/blob/master/configs/README.md).
