# configs/

Configuration lives here. Treat config files as **inputs** to pipelines, not as a dumping ground.

## Typical contents
- `config.yaml`: bucket size, timezone, smoothing policy, output dirs
- `features_v1.yaml`: feature selection and parameters
- `model_lgbm.yaml`: LightGBM hyperparameters
- `labels_v1.yaml`: allowed labels + validation rules

## Invariants
- Changes to feature selection must bump schema hash (automatically computed).
- Keep defaults sane; avoid environment-specific paths here.
