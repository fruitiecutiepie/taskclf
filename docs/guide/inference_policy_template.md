# Inference policy template

taskclf resolves online and batch inference from `models/inference_policy.json` when present (see [core.inference_policy](../api/core/inference_policy.md)).

## Unlike `config.toml`

Per-user `config.toml` next to your processed data directory is written once on first run with every supported key (see [User config template](config_template.md)).

**`inference_policy.json` is not auto-created on startup.** It is created only when you explicitly ask for it—for example **Edit Inference Policy** in the tray when a real model bundle can seed it, or `taskclf policy create` / `taskclf train tune-reject --write-policy`.

## Review the canonical example

The repository holds a stable, checked-in copy of the placeholder starter shape (including inline `_help` and links to this guide):

| | URL |
|---|---|
| **Browse on GitHub** (default branch) | [github.com/fruitiecutiepie/taskclf/blob/master/configs/inference_policy.template.json](https://github.com/fruitiecutiepie/taskclf/blob/master/configs/inference_policy.template.json) |
| **Raw file** (for curl/wget or “Save link as…”) | [raw.githubusercontent.com/fruitiecutiepie/taskclf/master/configs/inference_policy.template.json](https://raw.githubusercontent.com/fruitiecutiepie/taskclf/master/configs/inference_policy.template.json) |

Authoritative API: [core.inference_policy](../api/core/inference_policy.md) (`render_default_inference_policy_template_json`).

## What updates the live file

- **Edit Inference Policy** — opens the live policy when it exists, or seeds one from a resolved model bundle when possible. If no model can be resolved, it notifies you to use the CLI instead of writing a placeholder file.
- **CLI** — `taskclf policy create`, `taskclf policy remove`, or tuning flows that write a policy.

The app does **not** rewrite `inference_policy.json` on every launch to match defaults.

## Relation to pipeline configs

Files under `configs/` for training (for example `config.yml`) are not the same as deployment `models/inference_policy.json`. See [configs/README](https://github.com/fruitiecutiepie/taskclf/blob/master/configs/README.md).
