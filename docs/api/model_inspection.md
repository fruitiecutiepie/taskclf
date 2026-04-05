# model_inspection

Read-only inspection of trained model bundles and optional replay of held-out
test evaluation.

## Bundle vs replay

- **Bundle artifacts** (`metrics.json`, `confusion_matrix.csv`) in
  `models/<run_id>/` record **validation** metrics from
  :func:`~taskclf.train.lgbm.train_lgbm`, not the chronological test split.
  From the saved confusion matrix alone, inspection can derive **per-class
  support**, **per-class precision/recall/F1**, and **top confusion pairs**
  (largest off-diagonal counts).  Probability-based scores (ECE, Brier, log
  loss), **slice metrics**, and **unknown-category rates** are **not**
  persisted in the bundle; they need a dataframe replay.
- **Held-out test** evaluation replays the same pipeline as
  :func:`~taskclf.train.evaluate.evaluate_model` on labeled data for a date
  range (see :func:`~taskclf.model_inspection.replay_test_evaluation`).  That
  path fills `replayed_test_evaluation.report` with a full
  :class:`~taskclf.train.evaluate.EvaluationReport` (including calibration
  scalars, slices, unknown rates, and test **class distribution**).  The same
  enriched report is written to `evaluation.json` by `train evaluate`.

## Public API

::: taskclf.model_inspection
