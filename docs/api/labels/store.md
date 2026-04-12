# labels.store

Label span I/O, validation, and synthetic label generation.

## Key functions

| Function | Description |
|----------|-------------|
| `write_label_spans` | Serialize spans to parquet |
| `read_label_spans` | Deserialize spans from parquet (converts `NaN` values to `None` for nullable fields) |
| `import_labels_from_csv` | Read spans from CSV (supports optional `user_id` and `confidence` columns) |
| `append_label_span` | Append a single span to an existing parquet file with overlap validation; if the previous same-user label has `extend_forward=True`, its `end_ts` is stretched to the new span's `start_ts` for contiguous coverage. Before overlap checks, same-user boundaries within 1 ms are snapped together so JavaScript millisecond precision does not create false microsecond overlaps. Accepts `allow_overlap=True` to skip the overlap check and permit multiple labels on the same time range |
| `overwrite_label_span` | Append a span, resolving overlaps by truncating, splitting, or removing conflicting same-user spans. For `extend_forward` labels, overlap resolution uses the label's effective coverage through the next same-user label (or open-endedly when none exists), so retrospective inserts can split a running label into before/after fragments and preserve the resumed active fragment |
| `update_label_span` | Change the label, timestamps, and optionally `extend_forward` on an existing span identified by its `start_ts` and `end_ts`; validates the resulting span against `LABEL_SET_V1` |
| `delete_label_span` | Remove a label span identified by its `start_ts` and `end_ts` |
| `generate_label_summary` | Summarise features in a time range (top apps, input rates, session count) |
| `generate_dummy_labels` | Create synthetic spans for testing |

::: taskclf.labels.store
