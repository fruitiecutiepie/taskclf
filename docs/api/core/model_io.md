# core.model_io

Model bundle persistence: save, load, and metadata for trained model artifacts.

## ModelMetadata Fields

| Field | Type | Description |
|-------|------|-------------|
| `schema_version` | str | Feature schema version (e.g. `"v1"`) |
| `schema_hash` | str | Deterministic hash of the feature schema |
| `label_set` | list[str] | Sorted list of core labels used in training |
| `train_date_from` | str | First date of the training range (ISO-8601) |
| `train_date_to` | str | Last date of the training range (ISO-8601) |
| `params` | dict | Model hyperparameters |
| `git_commit` | str | Git commit SHA at training time |
| `dataset_hash` | str | SHA-256 hash of the training dataset for reproducibility |
| `reject_threshold` | float or None | Reject threshold used during evaluation |
| `data_provenance` | str | Origin: `"real"`, `"synthetic"`, or `"mixed"` |
| `created_at` | str | ISO-8601 timestamp of bundle creation |

::: taskclf.core.model_io
