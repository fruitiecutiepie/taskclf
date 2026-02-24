# train.build_dataset

Training dataset builder: join features with labels, apply exclusion
rules, split by time, and write `X.parquet`, `y.parquet`, and
`splits.json`.

## Usage

```python
from pathlib import Path
from taskclf.train.build_dataset import build_training_dataset

manifest = build_training_dataset(
    features_df,
    label_spans,
    output_dir=Path("data/processed/training_dataset"),
    train_ratio=0.70,
    val_ratio=0.15,
    holdout_user_fraction=0.1,
)
print(manifest.total_rows, manifest.train_rows)
```

## Output artifacts

| File | Contents |
|------|----------|
| `X.parquet` | Feature columns + ID columns (`user_id`, `bucket_start_ts`, `session_id`) + `schema_version` |
| `y.parquet` | `user_id`, `bucket_start_ts`, `label`, `provenance` |
| `splits.json` | Train/val/test index lists, holdout users, and metadata (schema versions, class distribution, user count) |

## Exclusion rules

Windows are dropped from the dataset if:

- They overlap multiple label blocks or have no covering label (handled by `assign_labels_to_buckets`).
- All numeric features are null (no useful signal).
- They belong to sessions shorter than `MIN_BLOCK_DURATION_SECONDS` (180s = 3 buckets).

::: taskclf.train.build_dataset
