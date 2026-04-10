"""Read-only inspection of trained model bundles and optional test-set replay.

Bundle ``metrics.json`` / ``confusion_matrix.csv`` reflect **validation** metrics
from training (see :func:`~taskclf.train.lgbm.train_lgbm`). Held-out **test**
metrics and class distribution require replaying evaluation on labeled data for
a date range (same pipeline as ``taskclf train evaluate``).
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, Field

from taskclf.core.defaults import DEFAULT_REJECT_THRESHOLD
from taskclf.core.metrics import class_distribution, top_confusion_pairs
from taskclf.core.model_io import ModelMetadata, load_model_bundle
from taskclf.core.schema import resolve_feature_parquet_path
from taskclf.core.types import LABEL_SET_V1
from taskclf.features.build import generate_dummy_features
from taskclf.labels.projection import project_blocks_to_windows
from taskclf.labels.store import generate_dummy_labels, read_label_spans
from taskclf.core.store import read_parquet
from taskclf.train.dataset import split_by_time
from taskclf.train.evaluate import evaluate_model


def per_class_metrics_from_confusion_matrix(
    cm: list[list[int]],
    label_names: list[str],
) -> dict[str, dict[str, float | int]]:
    """Derive per-class precision, recall, and F1 from a square confusion matrix.

    Uses the same layout as :func:`sklearn.metrics.confusion_matrix` with
    ``labels=label_names``: rows are **true** class, columns are **predicted**
    class.

    Args:
        cm: Square matrix ``len(label_names) x len(label_names)``.
        label_names: Ordered class names (row/column order).

    Returns:
        Mapping each label to ``precision``, ``recall``, and ``f1`` (rounded
        to 4 decimals, matching :func:`~taskclf.core.metrics.per_class_metrics`).
    """
    n = len(label_names)
    if len(cm) != n or any(len(row) != n for row in cm):
        raise ValueError(f"confusion matrix must be {n}x{n}, got {len(cm)} rows")
    result: dict[str, dict[str, float | int]] = {}
    for i, name in enumerate(label_names):
        row_sum = sum(cm[i][j] for j in range(n))
        col_sum = sum(cm[j][i] for j in range(n))
        tp = int(cm[i][i])
        prec = float(tp / col_sum) if col_sum > 0 else 0.0
        rec = float(tp / row_sum) if row_sum > 0 else 0.0
        if prec + rec > 0:
            f1 = 2.0 * prec * rec / (prec + rec)
        else:
            f1 = 0.0
        result[name] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "support": int(row_sum),
        }
    return result


def _load_label_spans_for_range(
    data_dir: Path,
    start: dt.date,
    end: dt.date,
) -> list:
    """Load label spans from disk, filtered to the given date range."""
    labels_path = data_dir / "labels_v1" / "labels.parquet"
    if not labels_path.exists():
        return []

    all_spans = read_label_spans(labels_path)
    start_dt = dt.datetime(start.year, start.month, start.day, tzinfo=dt.timezone.utc)
    end_dt = dt.datetime(
        end.year, end.month, end.day, 23, 59, 59, tzinfo=dt.timezone.utc
    )
    return [s for s in all_spans if s.end_ts >= start_dt and s.start_ts <= end_dt]


def build_labeled_dataframe(
    date_from: dt.date,
    date_to: dt.date,
    *,
    data_dir: Path,
    synthetic: bool,
) -> pd.DataFrame:
    """Load features and labels for ``[date_from, date_to]``, project to windows.

    Raises:
        ValueError: If no feature rows exist or the labeled frame is empty.
    """
    all_features: list[pd.DataFrame] = []
    all_labels: list = []
    current = date_from

    if not synthetic:
        all_labels = _load_label_spans_for_range(data_dir, date_from, date_to)

    while current <= date_to:
        if synthetic:
            rows = generate_dummy_features(current, n_rows=60)
            df = pd.DataFrame([r.model_dump() for r in rows])
            labels = generate_dummy_labels(current, n_rows=60)
            all_labels.extend(labels)
        else:
            parquet_path = resolve_feature_parquet_path(data_dir, current)
            if parquet_path is None:
                current += dt.timedelta(days=1)
                continue
            df = read_parquet(parquet_path)
        all_features.append(df)
        current += dt.timedelta(days=1)

    if not all_features:
        raise ValueError(
            "No feature data found for the given date range (check --data-dir "
            "and feature parquet paths)."
        )

    features_df = pd.concat(all_features, ignore_index=True)
    labeled_df = project_blocks_to_windows(features_df, all_labels)
    if labeled_df.empty:
        raise ValueError(
            "No labeled rows after projection — cannot replay test evaluation."
        )
    return labeled_df


class PredictionLogicInfo(BaseModel, frozen=True):
    """Stable description of how multiclass predictions are formed."""

    problem_type: Literal["multiclass"] = "multiclass"
    multilabel: Literal[False] = False
    lightgbm_outputs: str = (
        "Raw class probabilities: np.asarray(model.predict(X)) with shape "
        "(n_rows, n_classes) for objective multiclass."
    )
    argmax_rule: str = (
        "Predicted class index: proba.argmax(axis=1); label strings via "
        "LabelEncoder.inverse_transform (canonical vocabulary LABEL_SET_V1 in "
        "evaluate_model; training uses the same sorted label order)."
    )
    evaluation_reject: str = (
        "When evaluating with reject_threshold, low-confidence rows may be "
        "mapped to Mixed/Unknown before metrics; reject_rate reports that fraction."
    )
    code_references: dict[str, str] = Field(
        default_factory=lambda: {
            "predict_proba": "taskclf.infer.batch.predict_proba",
            "train_eval_argmax": "taskclf.train.lgbm.train_lgbm",
            "evaluate_model": "taskclf.train.evaluate.evaluate_model",
            "metrics": "taskclf.core.metrics",
        }
    )


def prediction_logic_info() -> PredictionLogicInfo:
    """Return a frozen description of prediction and metric code paths."""
    return PredictionLogicInfo()


class BundleInspectionSection(BaseModel, frozen=True):
    """Metrics persisted in the bundle at train time (validation split)."""

    source: Literal["bundle_saved_validation"] = "bundle_saved_validation"
    description: str = (
        "macro_f1 / weighted_f1 / confusion_matrix in metrics.json are computed "
        "on the **validation** split inside train_lgbm, not the held-out test split."
    )
    macro_f1: float
    weighted_f1: float
    label_names: list[str]
    confusion_matrix: list[list[int]]
    per_class_derived: dict[str, dict[str, float | int]]
    top_confusion_pairs: list[dict[str, str | int]] = Field(default_factory=list)


class ReplayTestSection(BaseModel, frozen=True):
    """Held-out test replay (same pipeline as taskclf train evaluate)."""

    source: Literal["replayed_test_evaluation"] = "replayed_test_evaluation"
    test_row_count: int
    holdout_users: list[str]
    date_from: str
    date_to: str
    data_dir: str
    synthetic: bool
    holdout_fraction: float
    reject_threshold: float
    test_class_distribution: dict[str, dict[str, float | int]]
    report: dict[str, Any]


class ModelInspectResult(BaseModel, frozen=True):
    """Complete output of :func:`inspect_model`."""

    bundle_path: str
    metadata: dict[str, Any]
    bundle_saved_validation: BundleInspectionSection
    prediction_logic: PredictionLogicInfo
    replayed_test_evaluation: ReplayTestSection | None = None
    replay_error: str | None = None


def inspect_bundle_only(
    model_dir: Path | str,
) -> tuple[Path, ModelMetadata, BundleInspectionSection]:
    """Load bundle and build the validation-metrics inspection section.

    Returns:
        Resolved bundle path, metadata, and bundle_saved_validation section.
    """
    path = Path(model_dir).resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Model bundle directory not found: {path}")

    _, metadata, _ = load_model_bundle(path)

    raw_metrics = json.loads((path / "metrics.json").read_text())
    macro_f1 = float(raw_metrics["macro_f1"])
    weighted_f1 = float(raw_metrics["weighted_f1"])
    cm = raw_metrics["confusion_matrix"]
    label_names = list(raw_metrics["label_names"])
    per_class = per_class_metrics_from_confusion_matrix(cm, label_names)
    pairs = top_confusion_pairs(cm, label_names)

    section = BundleInspectionSection(
        macro_f1=macro_f1,
        weighted_f1=weighted_f1,
        label_names=label_names,
        confusion_matrix=cm,
        per_class_derived=per_class,
        top_confusion_pairs=pairs,
    )
    return path, metadata, section


def replay_test_evaluation(
    model_dir: Path | str,
    date_from: dt.date,
    date_to: dt.date,
    *,
    data_dir: Path,
    synthetic: bool = False,
    holdout_fraction: float = 0.0,
    reject_threshold: float = DEFAULT_REJECT_THRESHOLD,
) -> ReplayTestSection:
    """Run held-out evaluation on the test split (same as ``train evaluate``)."""
    bundle_path = Path(model_dir).resolve()
    model, metadata, cat_encoders = load_model_bundle(bundle_path)

    labeled_df = build_labeled_dataframe(
        date_from, date_to, data_dir=data_dir, synthetic=synthetic
    )
    splits = split_by_time(labeled_df, holdout_user_fraction=holdout_fraction)
    test_df = labeled_df.iloc[splits["test"]].reset_index(drop=True)
    holdout_users = splits.get("holdout_users", [])

    if test_df.empty:
        raise ValueError("Test set is empty — cannot replay evaluation.")

    report = evaluate_model(
        model,
        test_df,
        cat_encoders=cat_encoders,
        holdout_users=holdout_users,
        reject_threshold=reject_threshold,
        schema_version=metadata.schema_version,
    )

    label_order = sorted(LABEL_SET_V1)
    dist = class_distribution(list(test_df["label"].values), label_order)

    return ReplayTestSection(
        test_row_count=len(test_df),
        holdout_users=list(holdout_users),
        date_from=date_from.isoformat(),
        date_to=date_to.isoformat(),
        data_dir=str(data_dir.resolve()),
        synthetic=synthetic,
        holdout_fraction=holdout_fraction,
        reject_threshold=reject_threshold,
        test_class_distribution=dist,
        report=report.model_dump(),
    )


def inspect_model(
    model_dir: Path | str,
    *,
    date_from: dt.date | None = None,
    date_to: dt.date | None = None,
    data_dir: Path | None = None,
    synthetic: bool = False,
    holdout_fraction: float = 0.0,
    reject_threshold: float = DEFAULT_REJECT_THRESHOLD,
) -> ModelInspectResult:
    """Inspect a bundle; optionally replay test evaluation when dates are set.

    When *date_from* and *date_to* are both provided, *data_dir* must be set
    (or use synthetic data with ``synthetic=True``).
    """
    path, metadata, bundle_section = inspect_bundle_only(model_dir)

    replayed: ReplayTestSection | None = None
    replay_error: str | None = None

    if date_from is not None and date_to is not None:
        if data_dir is None and not synthetic:
            replay_error = (
                "Replay requires --data-dir when not using --synthetic, or pass "
                "synthetic=True."
            )
        else:
            try:
                dd = data_dir if data_dir is not None else Path(".")
                replayed = replay_test_evaluation(
                    path,
                    date_from,
                    date_to,
                    data_dir=dd,
                    synthetic=synthetic,
                    holdout_fraction=holdout_fraction,
                    reject_threshold=reject_threshold,
                )
            except (ValueError, FileNotFoundError, OSError) as exc:
                replay_error = str(exc)

    return ModelInspectResult(
        bundle_path=str(path),
        metadata=metadata.model_dump(),
        bundle_saved_validation=bundle_section,
        prediction_logic=prediction_logic_info(),
        replayed_test_evaluation=replayed,
        replay_error=replay_error,
    )


__all__ = [
    "BundleInspectionSection",
    "ModelInspectResult",
    "PredictionLogicInfo",
    "ReplayTestSection",
    "build_labeled_dataframe",
    "inspect_bundle_only",
    "inspect_model",
    "per_class_metrics_from_confusion_matrix",
    "prediction_logic_info",
    "replay_test_evaluation",
]
