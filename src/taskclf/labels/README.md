# labels/

Label handling using **time spans**.

## Modules
- `store.py` — Import/export label spans (CSV/parquet), append individual spans,
  generate label summaries and dummy labels for testing.
- `projection.py` — Project label blocks onto feature windows using strict
  containment rules (a window gets a label only if the span fully covers it).
- `queue.py` — Active labeling queue: surfaces low-confidence or drift-flagged
  windows for human review, tracks pending/completed requests.
- `weak_rules.py` — Optional heuristic rules that emit low-confidence label spans.

## Responsibilities
- Import/export label spans (CSV/parquet)
- Validate label set and time ranges
- Project label blocks onto per-minute feature windows for training
- Manage an active labeling queue for human-in-the-loop review
- Support real-time labeling: the CLI `label-now` command and the web UI
  "Recent" tab create spans from `now - N minutes` to `now`, with
  optional live ActivityWatch summaries
- Auto-enqueue low-confidence predictions from the online inference loop
- Optional weak labeling rules (heuristics) that emit low-confidence spans

## Invariants
- Labels are spans: (start_ts, end_ts, label, provenance).
- Label vocabulary is versioned (labels_v1).
- Never mix weak labels into gold labels without marking provenance.
