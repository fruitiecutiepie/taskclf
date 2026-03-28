# core.model_io

Model bundle persistence: save, load, and metadata for trained model artifacts.

## ModelMetadata Fields

| Field | Type | Description |
|-------|------|-------------|
| `schema_version` | str | Feature schema version (`"v1"` or `"v2"`) |
| `schema_hash` | str | Deterministic hash of the feature schema |
| `label_set` | list[str] | Sorted list of core labels used in training |
| `train_date_from` | str | First date of the training range (ISO-8601) |
| `train_date_to` | str | Last date of the training range (ISO-8601) |
| `params` | dict | Model hyperparameters |
| `git_commit` | str | Git commit SHA at training time |
| `dataset_hash` | str | SHA-256 hash of the training dataset for reproducibility |
| `reject_threshold` | float or None | Reject threshold used during evaluation. **Advisory only** — the canonical runtime threshold lives in [`InferencePolicy`](inference_policy.md). |
| `data_provenance` | str | Origin: `"real"`, `"synthetic"`, or `"mixed"` |
| `created_at` | str | ISO-8601 timestamp of bundle creation |
| `unknown_category_freq_threshold` | int or None | Minimum category frequency used during training (categories below this become `__unknown__`) |
| `unknown_category_mask_rate` | float or None | Fraction of known categories randomly masked to `__unknown__` during training |

## Schema Version Support

`load_model_bundle` validates bundles against a registry of known schema
versions.  Both `v1` and `v2` bundles are accepted; the bundle's
`schema_version` field selects the expected hash.  A hash mismatch
(e.g. loading a v1 bundle whose hash has been tampered to v2's value)
raises `ValueError`.

`build_metadata` accepts a `schema_version` parameter (default `"v1"`)
and fills in the correct version string and hash automatically.

::: taskclf.core.model_io
