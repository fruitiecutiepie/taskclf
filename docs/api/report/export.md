# report.export

Report export utilities with sensitive-field guards.

## Overview

Exports a [`DailyReport`](daily.md) to JSON, CSV, or Parquet.  All
three formats apply the same privacy guard: a recursive check rejects
any report containing sensitive keys before writing to disk.

## Sensitive-field redaction

Every export function calls `_check_no_sensitive_fields` before
writing.  The following keys are forbidden at any nesting depth:

| Key | Reason |
|-----|--------|
| `raw_keystrokes` | Raw keystroke data violates privacy policy |
| `window_title_raw` | Unhashed window titles may contain private information |
| `clipboard_content` | Clipboard contents may contain passwords or secrets |
| `clipboard` | Alias for clipboard data |

If any forbidden key is found (including in nested dicts), a
`ValueError` is raised and no file is written.

## Output formats

### JSON (`export_report_json`)

```python
export_report_json(report: DailyReport, path: Path) -> Path
```

Serializes the report via `model_dump(exclude_none=True)` and writes
formatted JSON (2-space indent).  `None`-valued optional fields
(`mapped_breakdown`, `context_switch_stats`, `flap_rate_raw`,
`flap_rate_smoothed`) are omitted from the output.

Creates parent directories if they do not exist.  Returns the written
path.

### CSV (`export_report_csv`)

```python
export_report_csv(report: DailyReport, path: Path) -> Path
```

Flattens the report's `core_breakdown` and `mapped_breakdown` into
tabular rows with one row per label:

| Column | Description |
|--------|-------------|
| `date` | Calendar date from the report |
| `label_type` | `core` or `mapped` |
| `label` | Label name |
| `minutes` | Minutes rounded to 2 decimal places |

Core rows are sorted alphabetically by label name, followed by mapped
rows (also sorted).  Creates parent directories if needed.

### Parquet (`export_report_parquet`)

```python
export_report_parquet(report: DailyReport, path: Path) -> Path
```

Same schema as CSV (`date`, `label_type`, `label`, `minutes`) written
as a Parquet file via pandas.  Creates parent directories if needed.

## Usage

```python
from pathlib import Path
from taskclf.report.daily import build_daily_report
from taskclf.report.export import (
    export_report_json,
    export_report_csv,
    export_report_parquet,
)

report = build_daily_report(segments)

export_report_json(report, Path("artifacts/report.json"))
export_report_csv(report, Path("artifacts/report.csv"))
export_report_parquet(report, Path("artifacts/report.parquet"))
```

All three functions raise `ValueError` if the serialized report
contains any key from the sensitive-fields blocklist.

See [`report.daily`](daily.md) for `DailyReport` construction.

::: taskclf.report.export
