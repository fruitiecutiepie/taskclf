# configs/

Configuration lives here. Treat config files as **inputs** to pipelines, not as a dumping ground.

## Contents
- `config.yml` — bucket size, title hashing salt, feature schema version, model params, split policy
- `retrain.yaml` — retraining workflow config (cadence, lookback, regression tolerance, training params)
- `user_taxonomy_example.yaml` — example user taxonomy mapping (core labels to user-facing buckets)
- `user_config.template.toml` — **example** of the per-install `config.toml` written next to `data/processed` (tray/UI user settings). Canonical copy for review; runtime file is created on first run with the same leading comment block pointing here. [Browse](https://github.com/fruitiecutiepie/taskclf/blob/master/configs/user_config.template.toml) · [raw](https://raw.githubusercontent.com/fruitiecutiepie/taskclf/master/configs/user_config.template.toml) · [docs](https://fruitiecutiepie.github.io/taskclf/guide/config_template/).
- `inference_policy.template.json` — **example** of the placeholder `models/inference_policy.json` starter (not auto-written on startup; created when editing policy or via CLI). Matches `render_default_inference_policy_template_json()` in code. [Browse](https://github.com/fruitiecutiepie/taskclf/blob/master/configs/inference_policy.template.json) · [raw](https://raw.githubusercontent.com/fruitiecutiepie/taskclf/master/configs/inference_policy.template.json) · [docs](https://fruitiecutiepie.github.io/taskclf/guide/inference_policy_template/).

## Invariants
- Changes to feature selection must bump schema hash (automatically computed).
- Keep defaults sane; avoid environment-specific paths here.
