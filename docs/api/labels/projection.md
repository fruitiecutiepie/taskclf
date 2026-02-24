# labels.projection

Block-to-window label projection following `time_spec.md` Section 6.

## Projection rules

1. A window is labeled only when its **entire** `[bucket_start_ts, bucket_end_ts)` falls within a single labeled block.
2. Windows covered by multiple blocks with **conflicting** labels are **dropped**.
3. Windows that only **partially** overlap a block are **dropped**.
4. Unlabeled windows are **dropped** (not used in supervised training).
5. When a span carries a `user_id`, it only matches features with the same `user_id`.

## Usage

```python
from taskclf.labels.projection import project_blocks_to_windows

projected_df = project_blocks_to_windows(features_df, label_spans)
```

The returned DataFrame contains only the cleanly-labeled windows, with an added `label` column.

::: taskclf.labels.projection
