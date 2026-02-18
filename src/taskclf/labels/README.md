# labels/

Label handling using **time spans**.

## Responsibilities
- Import/export label spans (CSV/parquet)
- Validate label set and time ranges
- Optional weak labeling rules (heuristics) that emit low-confidence spans

## Invariants
- Labels are spans: (start_ts, end_ts, label).
- Label vocabulary is versioned (labels_v1).
- Never mix weak labels into gold labels without marking provenance.
