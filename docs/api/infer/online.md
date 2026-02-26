# infer.online

Real-time prediction loop: poll ActivityWatch, predict, smooth, and report.

## Model auto-resolution

`--model-dir` is optional.  When omitted, the model is resolved
automatically using `resolve_model_dir()` from `taskclf.infer.resolve`:

1. If `models/active.json` exists and points to a valid, compatible bundle, use it.
2. Otherwise, scan `--models-dir` (default `models/`) and select the best bundle by policy.
3. If no eligible model is found, exit with a descriptive error.

`--model-dir` always overrides auto-resolution when provided.

## Model hot-reload

When `--models-dir` is provided (always the case with the default CLI),
the online loop watches `models/active.json` for changes using mtime
polling (default interval: 60 seconds).  When a change is detected:

1. The new active bundle is resolved and loaded.
2. The `OnlinePredictor` is rebuilt with the new model.
3. If loading fails, the current model is kept and a warning is logged.

This allows retraining to promote a new model while the online loop is
running, without requiring a restart.

## Label queue integration

When `label_queue_path` is provided, the online loop auto-enqueues
predictions whose confidence falls below `label_confidence_threshold`
(default 0.55) into the `ActiveLabelingQueue`.  Enqueued items surface
in `taskclf labels show-queue` and the Streamlit labeling UI for
manual review.

Enable via CLI:

```bash
taskclf infer online \
  --label-queue \
  --label-confidence 0.50
```

At shutdown, the loop prints how many buckets were enqueued during the session.

::: taskclf.infer.online

::: taskclf.infer.resolve
