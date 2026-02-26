# infer.online

Real-time prediction loop: poll ActivityWatch, predict, smooth, and report.

## Label queue integration

When `label_queue_path` is provided, the online loop auto-enqueues
predictions whose confidence falls below `label_confidence_threshold`
(default 0.55) into the `ActiveLabelingQueue`.  Enqueued items surface
in `taskclf labels show-queue` and the Streamlit labeling UI for
manual review.

Enable via CLI:

```bash
taskclf infer online \
  --model-dir models/run_... \
  --label-queue \
  --label-confidence 0.50
```

At shutdown, the loop prints how many buckets were enqueued during the session.

::: taskclf.infer.online
