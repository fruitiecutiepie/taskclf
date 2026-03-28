# infer.batch

Batch inference: predict, smooth, and segmentize over a feature DataFrame.

See [Inference Contract](../../guide/inference_contract.md) for the
canonical pipeline order and how this module fits the batch runtime path.

## Overview

`run_batch_inference` executes the full post-training prediction
pipeline in a single call:

```
predict_proba → calibrate → reject → smooth → segmentize → hysteresis merge
```

Each step is also available as a standalone function for granular use.

## BatchInferenceResult

Frozen dataclass returned by `run_batch_inference`.  Core prediction
fields are always populated; taxonomy-mapped fields are `None` when
no taxonomy config was provided.

| Field | Type | Description |
|-------|------|-------------|
| `raw_labels` | `list[str]` | Pre-smoothing labels (rejected buckets become `Mixed/Unknown`) |
| `smoothed_labels` | `list[str]` | Post-smoothing labels |
| `segments` | `list[Segment]` | Hysteresis-merged contiguous segments |
| `confidences` | `np.ndarray` | `max(proba)` per row |
| `is_rejected` | `np.ndarray` | Boolean rejection flags |
| `core_probs` | `np.ndarray` | `(N, 8)` probability matrix |
| `mapped_labels` | `list[str] \| None` | Taxonomy-mapped labels (if taxonomy provided) |
| `mapped_probs` | `list[dict] \| None` | Per-bucket mapped probabilities (if taxonomy provided) |

## Functions

### predict_proba

Return the raw probability matrix `(n_rows, n_classes)` for a feature
DataFrame.  Applies categorical encoding via `encode_categoricals`
before prediction.

### predict_labels

Run the model and return predicted label strings.  When
`reject_threshold` is set, predictions with `max(proba)` below the
threshold are replaced with `Mixed/Unknown`.

### run_batch_inference

End-to-end pipeline that predicts, calibrates, rejects, smooths,
segmentizes, and applies hysteresis merging.

Optional parameters:

- `calibrator` -- a single `Calibrator` instance applied to all rows.
- `calibrator_store` -- a `CalibratorStore` for per-user calibration;
  takes precedence over `calibrator` when `user_id` is present in the
  DataFrame.
- `taxonomy` -- a `TaxonomyConfig` to map core labels to user-defined
  buckets, populating `mapped_labels` and `mapped_probs` on the result.
- `reject_threshold` -- confidence floor; predictions below this become
  `Mixed/Unknown` before smoothing.
- `smooth_window` -- window size for `rolling_majority` (default from
  `core.defaults`).

### write_predictions_csv

Write per-bucket predictions to CSV with columns for timestamp,
predicted label, confidence, rejection flag, mapped label, and
core probabilities.

### write_segments_json / read_segments_json

Serialize and deserialize `Segment` lists as JSON.  Timestamps are
stored in ISO 8601 format.

## Usage

```python
from pathlib import Path
from taskclf.infer.batch import run_batch_inference, write_segments_json

result = run_batch_inference(
    model, features_df,
    cat_encoders=cat_encoders,
    reject_threshold=0.40,
)

print(f"{len(result.segments)} segments, "
      f"{result.is_rejected.sum()} rejected buckets")

write_segments_json(result.segments, Path("artifacts/segments.json"))
```

::: taskclf.infer.batch
