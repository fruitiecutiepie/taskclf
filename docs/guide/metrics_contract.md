# metrics.json contract

This document defines the stable contract for `metrics.json` stored inside each model bundle directory.

`metrics.json` is consumed by:
- registry scanning / ranking (best model selection)
- reporting commands (e.g., `taskclf train list`)
- debugging / inspection tooling

Acceptance checks (gates) are NOT in `metrics.json` today; they live in `evaluation.json` under the evaluation output directory (typically `artifacts/`).

## Location

Inside a model bundle directory (see `docs/model_bundle_layout.md`):
- `<bundle_dir>/metrics.json`

## Schema (current)

`metrics.json` is a single JSON object with exactly these keys:

```json
{
  "macro_f1": 0.0,
  "weighted_f1": 0.0,
  "confusion_matrix": [[0, 1], [2, 3]],
  "label_names": ["BreakIdle", "Coding"]
}
````

### Fields

* `macro_f1` (float)

  * Macro-averaged F1 across all classes.
  * Range: [0.0, 1.0].

* `weighted_f1` (float)

  * Support-weighted F1 across all classes.
  * Range: [0.0, 1.0].

* `confusion_matrix` (list[list[int]])

  * NxN confusion matrix.
  * N must equal `len(label_names)`.
  * Integers must be >= 0.
  * Convention should be stable; tooling assumes rows=true labels, cols=predicted labels unless explicitly documented elsewhere.

* `label_names` (list[str])

  * Label ordering corresponding to rows/cols in `confusion_matrix`.
  * Should match the runtime label set (LABEL_SET_V1) when the bundle is compatible.

## Non-goals / excluded today

The following are NOT stored in `metrics.json` today:

* per-class precision/recall/F1
* acceptance check booleans
* calibration artifacts
* per-user metrics
* reject rate

These appear in `evaluation.json` (written by evaluation artifacts code path to the evaluation output directory, not colocated with the bundle).

## Forward-compatible extension (recommended)

If you want selection policy to incorporate acceptance gates without depending on external `evaluation.json`, add one of:

A) Extend `metrics.json` with an optional `acceptance_checks` map:

```json
"acceptance_checks": {
  "macro_f1": true,
  "weighted_f1": true,
  "breakidle_precision": true,
  "breakidle_recall": true,
  "no_class_below_50_precision": true,
  "reject_rate_bounds": true
}
```

B) Add a colocated `acceptance.json` inside the bundle at promotion time.

If you extend the schema, keep the existing keys stable and additive only.
