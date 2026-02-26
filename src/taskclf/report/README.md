# report/

Reporting and exports.

## Modules
- `daily.py` — Build daily reports from segments: label breakdown, flap rates,
  mapped-label breakdown, context-switching statistics.
- `export.py` — Export reports as JSON, CSV, or Parquet.

## Responsibilities
- Convert predictions into daily totals by label
- Generate segment timelines
- Compute smoothing quality metrics (flap rates)
- Export JSON/CSV/Parquet summaries

## Invariants
- Reports are derived; never treated as source-of-truth.
- Prefer deterministic outputs given the same inputs.
