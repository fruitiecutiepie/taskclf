# labels.store

Label span I/O, validation, and synthetic label generation.

## Key functions

| Function | Description |
|----------|-------------|
| `write_label_spans` | Serialize spans to parquet |
| `read_label_spans` | Deserialize spans from parquet |
| `import_labels_from_csv` | Read spans from CSV (supports optional `user_id` and `confidence` columns) |
| `append_label_span` | Append a single span to an existing parquet file with overlap validation; if the previous same-user label has `extend_forward=True`, its `end_ts` is stretched to the new span's `start_ts` for contiguous coverage |
| `generate_label_summary` | Summarise features in a time range (top apps, input rates, session count) |
| `generate_dummy_labels` | Create synthetic spans for testing |

::: taskclf.labels.store
