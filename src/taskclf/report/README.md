# report/

Reporting and exports.

## Responsibilities
- Convert predictions into daily totals by label
- Generate segment timelines
- Export JSON/CSV/HTML summaries

## Invariants
- Reports are derived; never treated as source-of-truth.
- Prefer deterministic outputs given the same inputs.
