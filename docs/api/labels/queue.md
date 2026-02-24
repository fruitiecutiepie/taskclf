# labels.queue

Active labeling queue for prioritising windows that need human labels.

## Overview

The queue tracks buckets flagged for labeling due to low model confidence
or detected drift.  A daily ask limit prevents user fatigue.

## LabelRequest

Pydantic model representing a single labeling request:

| Field | Type | Description |
|-------|------|-------------|
| `request_id` | `str` | UUID |
| `user_id` | `str` | User whose bucket needs labeling |
| `bucket_start_ts` | `datetime` | Bucket start (UTC) |
| `bucket_end_ts` | `datetime` | Bucket end (UTC, exclusive) |
| `reason` | `"low_confidence" \| "drift"` | Why enqueued |
| `confidence` | `float \| None` | Model confidence at enqueue time |
| `predicted_label` | `str \| None` | Model prediction at enqueue time |
| `created_at` | `datetime` | Creation timestamp (UTC) |
| `status` | `"pending" \| "labeled" \| "skipped"` | Lifecycle state |

## ActiveLabelingQueue

```python
from taskclf.labels.queue import ActiveLabelingQueue

queue = ActiveLabelingQueue(Path("data/processed/labels_v1/queue.json"))
queue.enqueue_low_confidence(predictions_df, threshold=0.55)
pending = queue.get_pending(user_id="u1", limit=10)
queue.mark_done(pending[0].request_id, status="labeled")
```

::: taskclf.labels.queue
