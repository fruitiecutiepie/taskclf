# infer.prediction

Structured prediction output for a single time-window inference.

## WindowPrediction

Frozen Pydantic model representing the full inference output for one
time bucket.  Matches the contract defined in
[model IO guide, section 6](../../guide/model_io.md).

Every field is populated regardless of whether a taxonomy or calibrator
is active; when no taxonomy is configured the mapped fields mirror the
core prediction.

### Fields

| Field | Type | Constraints | Description |
|-------|------|-------------|-------------|
| `user_id` | `str` | -- | User identifier |
| `bucket_start_ts` | `datetime` | -- | Bucket start time |
| `core_label_id` | `int` | `0--7` | Predicted core label index |
| `core_label_name` | `str` | -- | Predicted core label name |
| `core_probs` | `list[float]` | length 8, sums to 1.0 | Per-class probabilities |
| `confidence` | `float` | `[0.0, 1.0]` | `max(core_probs)` |
| `is_rejected` | `bool` | -- | Below reject threshold |
| `mapped_label_name` | `str` | -- | Taxonomy-mapped label |
| `mapped_probs` | `dict[str, float]` | sums to 1.0 | Bucket-level probabilities |
| `model_version` | `str` | -- | Schema hash of the model bundle |
| `schema_version` | `str` | runtime-populated | Feature schema version (`features_v1`, `features_v2`, or `features_v3`) |
| `label_version` | `str` | default `labels_v1` | Label schema version |

### Validation rules

- `core_probs` must sum to 1.0 (tolerance 1e-4).
- `mapped_probs` values must sum to 1.0 (tolerance 1e-4).
- The model is **frozen** (`frozen=True`): instances are immutable
  after creation.

Validation is enforced via a Pydantic `model_validator` that runs
automatically on construction.

::: taskclf.infer.prediction
