# model_inspection

Read-only inspection of trained model bundles and optional replay of held-out
test evaluation.

## Bundle vs replay

- **Bundle artifacts** (`metrics.json`, `confusion_matrix.csv`) in
  `models/<run_id>/` record **validation** metrics from
  :func:`~taskclf.train.lgbm.train_lgbm`, not the chronological test split.
- **Held-out test** metrics, per-class precision/recall/F1, reject rate, and
  **test-set class distribution** require replaying the same pipeline as
  :func:`~taskclf.train.evaluate.evaluate_model` on labeled data for a date
  range (see :func:`~taskclf.model_inspection.replay_test_evaluation`).

## Public API

::: taskclf.model_inspection
