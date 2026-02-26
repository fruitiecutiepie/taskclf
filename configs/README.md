# configs/

Configuration lives here. Treat config files as **inputs** to pipelines, not as a dumping ground.

## Contents
- `config.yml` — bucket size, title hashing salt, feature schema version, model params, split policy
- `retrain.yaml` — retraining workflow config (cadence, lookback, regression tolerance, training params)
- `user_taxonomy_example.yaml` — example user taxonomy mapping (core labels to user-facing buckets)

## Invariants
- Changes to feature selection must bump schema hash (automatically computed).
- Keep defaults sane; avoid environment-specific paths here.
