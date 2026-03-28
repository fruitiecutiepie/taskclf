# infer.online

Real-time prediction loop: poll ActivityWatch, predict, smooth, and report.

See [Inference Contract](../../guide/inference_contract.md) for the
canonical pipeline order and how this module fits the online runtime path.
`OnlinePredictor` is implemented as a slotted dataclass while preserving
its constructor shape (`model`, `metadata`, then keyword-only options).

## Model auto-resolution

`--model-dir` is optional.  When omitted, the model is resolved
automatically using `resolve_model_dir()` from `taskclf.infer.resolve`:

1. If `models/active.json` exists and points to a valid, compatible bundle, use it.
2. Otherwise, scan `--models-dir` (default `models/`) and select the best bundle by policy.
3. If no eligible model is found, exit with a descriptive error.

`--model-dir` always overrides auto-resolution when provided.

## Model hot-reload

When `--models-dir` is provided (always the case with the default CLI),
the online loop watches `models/active.json` (see [model bundle layout](../../guide/model_io.md)) for changes using mtime
polling (default interval: 60 seconds).  When a change is detected:

1. The new active bundle is resolved and loaded.
2. The `OnlinePredictor` is rebuilt with the new model.
3. If loading fails, the current model is kept and a warning is logged.

This allows retraining to promote a new model while the online loop is
running, without requiring a restart.

## Label queue integration

When `label_queue_path` is provided, the online loop auto-enqueues
predictions whose confidence falls below `label_confidence_threshold`
(default 0.55) into the [`ActiveLabelingQueue`](../labels/queue.md).  Enqueued items surface
in `taskclf labels show-queue` and the web UI for
manual review.

Enable via CLI:

```bash
taskclf infer online \
  --label-queue \
  --label-confidence 0.50
```

At shutdown, the loop prints how many buckets were enqueued during the session.

## Persistent feature state

The online loop creates an `OnlineFeatureState` (see
[`infer.feature_state`](feature_state.md)) that buffers recent
`FeatureRow` values across poll cycles.  After each row is built from
the current poll window, it is pushed into the state and the corrected
rolling aggregates (15-minute app switch counts, rolling means, deltas,
session length) are overlaid onto the row before prediction.  This
ensures features match the full history windows the model was trained on,
rather than being truncated to the narrow poll slice.

The feature state is preserved across model hot-reloads since it tracks
feature history, not model state.

## Missing-value handling

`OnlinePredictor._encode_value()` fills missing numeric values with `0.0`,
matching the training and batch inference paths which use `fillna(0)`.

## Per-user reject thresholds

`OnlinePredictor` accepts an optional `per_user_reject_thresholds` dict
mapping user IDs to individual reject thresholds.  When present and the
current row's `user_id` is found in the dict, the per-user threshold
overrides the global `reject_threshold` for the rejection decision.
Users not in the dict fall back to the global threshold.

Per-user thresholds are loaded from `InferencePolicy.per_user_reject_thresholds`
by `resolve_inference_config()` and threaded through `ResolvedInferenceConfig`.

## Unknown-category handling

When a categorical value is not found in the fitted encoder's vocabulary,
`_encode_value()` checks whether the encoder contains an `"__unknown__"`
class (present when the model was trained with `encode_categoricals`
using `min_category_freq` / `unknown_mask_rate`).  If so, the
`__unknown__` code is returned; otherwise `-1.0` is used as a legacy
fallback.  This ensures that models trained with explicit unknown-category
exposure produce calibrated confidence on novel inputs rather than
defaulting to an out-of-vocabulary sentinel the model never learned.

::: taskclf.infer.online

::: taskclf.infer.resolve
