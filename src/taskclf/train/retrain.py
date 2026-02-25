"""Retraining workflow: cadence scheduling, reproducible pipeline, regression gates.

Implements TODO 11 from the labelling roadmap:

* **Cadence** — :func:`check_retrain_due` / :func:`check_calibrator_update_due`
  determine whether a global retrain or per-user calibrator update should run.
* **Reproducibility** — :func:`compute_dataset_hash` produces a deterministic
  SHA-256 of the training data so each model bundle can be traced back to its
  exact dataset.  :class:`RetrainConfig` is loadable from a versioned YAML file.
* **Regression gates** — :func:`check_regression_gates` compares a challenger
  model against the current champion on key metrics (macro-F1, BreakIdle
  precision, per-class precision) and only promotes if no regression exceeds
  the configured tolerance.
* **Pipeline** — :func:`run_retrain_pipeline` orchestrates the full flow from
  data loading through training, evaluation, gate checking, and promotion.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal, Sequence

import pandas as pd
import yaml
from pydantic import BaseModel, Field

from taskclf.core.defaults import (
    DEFAULT_CALIBRATOR_UPDATE_CADENCE_DAYS,
    DEFAULT_DATA_LOOKBACK_DAYS,
    DEFAULT_MODELS_DIR,
    DEFAULT_NUM_BOOST_ROUND,
    DEFAULT_REGRESSION_TOLERANCE,
    DEFAULT_REJECT_THRESHOLD,
    DEFAULT_RETRAIN_CADENCE_DAYS,
)
from taskclf.core.model_io import (
    ModelMetadata,
    build_metadata,
    load_model_bundle,
    save_model_bundle,
)
from taskclf.core.types import LabelSpan
from taskclf.train.evaluate import EvaluationReport, evaluate_model

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class TrainParams(BaseModel):
    """Hyperparameters forwarded to :func:`~taskclf.train.lgbm.train_lgbm`."""

    num_boost_round: int = DEFAULT_NUM_BOOST_ROUND
    class_weight: Literal["balanced", "none"] = "balanced"


class RetrainConfig(BaseModel):
    """Cadence, gate thresholds, and training parameters for the retrain loop."""

    global_retrain_cadence_days: int = DEFAULT_RETRAIN_CADENCE_DAYS
    calibrator_update_cadence_days: int = DEFAULT_CALIBRATOR_UPDATE_CADENCE_DAYS
    data_lookback_days: int = DEFAULT_DATA_LOOKBACK_DAYS
    regression_tolerance: float = DEFAULT_REGRESSION_TOLERANCE
    require_baseline_improvement: bool = True
    auto_promote: bool = False
    train_params: TrainParams = Field(default_factory=TrainParams)


def load_retrain_config(path: Path) -> RetrainConfig:
    """Load a :class:`RetrainConfig` from a YAML file.

    Args:
        path: Path to a YAML config file whose keys match
            :class:`RetrainConfig` fields.

    Returns:
        A validated config instance.
    """
    raw = yaml.safe_load(path.read_text())
    return RetrainConfig.model_validate(raw)


# ---------------------------------------------------------------------------
# Dataset hashing
# ---------------------------------------------------------------------------


class DatasetSnapshot(BaseModel, frozen=True):
    """Immutable record of the dataset used for a training run."""

    dataset_hash: str
    row_count: int
    date_from: str
    date_to: str
    user_count: int
    class_distribution: dict[str, int]


def compute_dataset_hash(features_df: pd.DataFrame, labels: Sequence[LabelSpan]) -> str:
    """Compute a deterministic SHA-256 of the training data.

    The hash is built from a canonical JSON representation of:

    * Sorted feature column names
    * Feature values serialized row-by-row (sorted by primary key)
    * Label spans serialized in chronological order

    Args:
        features_df: Feature DataFrame (pre-projection).
        labels: Label spans used for projection.

    Returns:
        A hex-encoded SHA-256 digest (first 16 characters).
    """
    h = hashlib.sha256()

    cols = sorted(features_df.columns.tolist())
    h.update(json.dumps(cols, sort_keys=True).encode())

    sorted_df = features_df.sort_values(
        ["user_id", "bucket_start_ts"]
    ).reset_index(drop=True)
    h.update(sorted_df.to_csv(index=False).encode())

    sorted_spans = sorted(labels, key=lambda s: (s.start_ts, s.end_ts))
    for span in sorted_spans:
        h.update(span.model_dump_json().encode())

    return h.hexdigest()[:16]


# ---------------------------------------------------------------------------
# Cadence checks
# ---------------------------------------------------------------------------


def find_latest_model(models_dir: Path) -> Path | None:
    """Return the path to the most recently created model bundle.

    Scans ``models_dir`` for subdirectories containing ``metadata.json``
    and picks the one with the latest ``created_at`` timestamp.

    Args:
        models_dir: Base directory containing model run folders.

    Returns:
        Path to the latest run directory, or ``None`` if none found.
    """
    if not models_dir.is_dir():
        return None

    best_path: Path | None = None
    best_ts: str = ""

    for candidate in models_dir.iterdir():
        meta_path = candidate / "metadata.json"
        if not meta_path.is_file():
            continue
        try:
            raw = json.loads(meta_path.read_text())
            created = raw.get("created_at", "")
            if created > best_ts:
                best_ts = created
                best_path = candidate
        except (json.JSONDecodeError, OSError):
            continue

    return best_path


def check_retrain_due(
    models_dir: Path,
    cadence_days: int = DEFAULT_RETRAIN_CADENCE_DAYS,
) -> bool:
    """Check whether a global retrain is due based on the last model's age.

    Args:
        models_dir: Base directory containing model run folders.
        cadence_days: Maximum age in days before a retrain is needed.

    Returns:
        ``True`` if retrain is due (or no model exists).
    """
    latest = find_latest_model(models_dir)
    if latest is None:
        return True

    raw = json.loads((latest / "metadata.json").read_text())
    created_at = raw.get("created_at", "")
    if not created_at:
        return True

    try:
        model_ts = datetime.fromisoformat(created_at)
        if model_ts.tzinfo is None:
            model_ts = model_ts.replace(tzinfo=UTC)
        age = datetime.now(UTC) - model_ts
        return age >= timedelta(days=cadence_days)
    except (ValueError, TypeError):
        return True


def check_calibrator_update_due(
    calibrator_store_dir: Path,
    cadence_days: int = DEFAULT_CALIBRATOR_UPDATE_CADENCE_DAYS,
) -> bool:
    """Check whether a calibrator store update is due.

    Args:
        calibrator_store_dir: Directory containing the calibrator store
            (expected to have a ``store.json`` with a ``created_at`` field).
        cadence_days: Maximum age in days before an update is needed.

    Returns:
        ``True`` if update is due (or no store exists).
    """
    store_file = calibrator_store_dir / "store.json"
    if not store_file.is_file():
        return True

    try:
        raw = json.loads(store_file.read_text())
        created = raw.get("created_at", "")
        if not created:
            return True
        store_ts = datetime.fromisoformat(created)
        if store_ts.tzinfo is None:
            store_ts = store_ts.replace(tzinfo=UTC)
        return datetime.now(UTC) - store_ts >= timedelta(days=cadence_days)
    except (json.JSONDecodeError, ValueError, TypeError, OSError):
        return True


# ---------------------------------------------------------------------------
# Regression gates
# ---------------------------------------------------------------------------


class RegressionGate(BaseModel, frozen=True):
    """Result of a single regression gate check."""

    name: str
    passed: bool
    detail: str


class RegressionResult(BaseModel, frozen=True):
    """Aggregate result of all regression gates."""

    all_passed: bool
    gates: list[RegressionGate]


def check_regression_gates(
    champion_report: EvaluationReport,
    challenger_report: EvaluationReport,
    config: RetrainConfig,
) -> RegressionResult:
    """Compare challenger model against champion on key metrics.

    Gates:

    1. **macro_f1_no_regression** — challenger macro-F1 must be within
       ``regression_tolerance`` of the champion.
    2. **breakidle_precision** — challenger BreakIdle precision >= 0.95.
    3. **no_class_below_50_precision** — no class may have precision < 0.50.
    4. **challenger_acceptance** — all of the challenger's own acceptance
       checks must pass.

    Args:
        champion_report: Evaluation report of the current deployed model.
        challenger_report: Evaluation report of the newly trained model.
        config: Retrain configuration with tolerance settings.

    Returns:
        A :class:`RegressionResult` with per-gate pass/fail details.
    """
    gates: list[RegressionGate] = []

    # Gate 1: macro-F1 no regression
    delta = champion_report.macro_f1 - challenger_report.macro_f1
    passed = delta <= config.regression_tolerance
    gates.append(RegressionGate(
        name="macro_f1_no_regression",
        passed=passed,
        detail=(
            f"champion={champion_report.macro_f1:.4f} "
            f"challenger={challenger_report.macro_f1:.4f} "
            f"delta={delta:+.4f} tolerance={config.regression_tolerance}"
        ),
    ))

    # Gate 2: BreakIdle precision invariant
    bi = challenger_report.per_class.get("BreakIdle", {})
    bi_prec = bi.get("precision", 0.0)
    gates.append(RegressionGate(
        name="breakidle_precision",
        passed=bi_prec >= 0.95,
        detail=f"BreakIdle precision={bi_prec:.4f} (>= 0.95 required)",
    ))

    # Gate 3: no class below 0.50 precision
    low_classes = [
        (name, m["precision"])
        for name, m in challenger_report.per_class.items()
        if m.get("precision", 0.0) < 0.50
    ]
    gates.append(RegressionGate(
        name="no_class_below_50_precision",
        passed=len(low_classes) == 0,
        detail=(
            "all classes >= 0.50"
            if not low_classes
            else "FAIL: " + ", ".join(f"{n}={p:.4f}" for n, p in low_classes)
        ),
    ))

    # Gate 4: challenger passes its own acceptance checks
    all_acceptance = all(challenger_report.acceptance_checks.values())
    failed_checks = [
        k for k, v in challenger_report.acceptance_checks.items() if not v
    ]
    gates.append(RegressionGate(
        name="challenger_acceptance",
        passed=all_acceptance,
        detail=(
            "all acceptance checks passed"
            if all_acceptance
            else "FAIL: " + ", ".join(failed_checks)
        ),
    ))

    return RegressionResult(
        all_passed=all(g.passed for g in gates),
        gates=gates,
    )


# ---------------------------------------------------------------------------
# Retrain pipeline
# ---------------------------------------------------------------------------


class RetrainResult(BaseModel):
    """Output of :func:`run_retrain_pipeline`."""

    promoted: bool
    champion_macro_f1: float | None = None
    challenger_macro_f1: float
    regression: RegressionResult | None = None
    dataset_snapshot: DatasetSnapshot
    run_dir: str
    reason: str


def run_retrain_pipeline(
    config: RetrainConfig,
    features_df: pd.DataFrame,
    label_spans: Sequence[LabelSpan],
    *,
    models_dir: Path = Path(DEFAULT_MODELS_DIR),
    out_dir: Path = Path("artifacts"),
    force: bool = False,
    dry_run: bool = False,
    holdout_user_fraction: float = 0.0,
    reject_threshold: float = DEFAULT_REJECT_THRESHOLD,
    data_provenance: Literal["real", "synthetic", "mixed"] = "real",
) -> RetrainResult:
    """Run the full retrain pipeline: train, evaluate, gate, promote.

    Steps:

    1. Optionally check cadence (skipped if *force*).
    2. Compute dataset snapshot hash.
    3. Project labels, split, and train a challenger model.
    4. Evaluate the challenger.
    5. If a champion exists, run regression gates.
    6. Promote the challenger if all gates pass (skipped if *dry_run*).

    Args:
        config: Retrain configuration.
        features_df: Feature DataFrame covering the training range.
        label_spans: Label spans for projection.
        models_dir: Base directory for model bundles.
        out_dir: Directory for evaluation artifacts.
        force: Skip cadence check.
        dry_run: Evaluate without promoting.
        holdout_user_fraction: Fraction of users to hold out for test.
        reject_threshold: Reject threshold for evaluation.
        data_provenance: Origin of the training data.

    Returns:
        A :class:`RetrainResult` summarizing what happened.
    """
    from taskclf.labels.projection import project_blocks_to_windows
    from taskclf.train.dataset import split_by_time
    from taskclf.train.lgbm import train_lgbm

    if not force and not check_retrain_due(models_dir, config.global_retrain_cadence_days):
        dataset_hash = compute_dataset_hash(features_df, label_spans)
        return RetrainResult(
            promoted=False,
            challenger_macro_f1=0.0,
            dataset_snapshot=DatasetSnapshot(
                dataset_hash=dataset_hash,
                row_count=len(features_df),
                date_from=str(features_df["bucket_start_ts"].min()),
                date_to=str(features_df["bucket_start_ts"].max()),
                user_count=features_df["user_id"].nunique(),
                class_distribution={},
            ),
            run_dir="",
            reason="Retrain not due yet",
        )

    # Dataset snapshot
    dataset_hash = compute_dataset_hash(features_df, label_spans)

    labeled_df = project_blocks_to_windows(features_df, label_spans)
    if labeled_df.empty:
        raise ValueError("No labeled rows after projection — cannot retrain")

    class_dist = labeled_df["label"].value_counts().to_dict()
    ts_col = features_df["bucket_start_ts"]

    snapshot = DatasetSnapshot(
        dataset_hash=dataset_hash,
        row_count=len(labeled_df),
        date_from=str(ts_col.min()),
        date_to=str(ts_col.max()),
        user_count=labeled_df["user_id"].nunique(),
        class_distribution={str(k): int(v) for k, v in class_dist.items()},
    )

    # Split
    splits = split_by_time(
        labeled_df,
        holdout_user_fraction=holdout_user_fraction,
    )
    train_df = labeled_df.iloc[splits["train"]].reset_index(drop=True)
    val_df = labeled_df.iloc[splits["val"]].reset_index(drop=True)
    test_df = labeled_df.iloc[splits["test"]].reset_index(drop=True)
    holdout_users: list[str] = splits.get("holdout_users", [])

    if train_df.empty or val_df.empty:
        raise ValueError("Train or validation split is empty — need more data")

    # Train challenger
    logger.info("Training challenger model (%d train, %d val rows)", len(train_df), len(val_df))
    model, metrics, cm_df, params, cat_encoders = train_lgbm(
        train_df, val_df,
        num_boost_round=config.train_params.num_boost_round,
        class_weight=config.train_params.class_weight,
    )

    # Evaluate challenger
    eval_df = test_df if not test_df.empty else val_df
    challenger_report = evaluate_model(
        model, eval_df,
        cat_encoders=cat_encoders,
        holdout_users=holdout_users,
        reject_threshold=reject_threshold,
    )

    # Determine date range from features
    date_from = pd.Timestamp(ts_col.min()).date()
    date_to = pd.Timestamp(ts_col.max()).date()

    metadata = build_metadata(
        label_set=list(metrics["label_names"]),
        train_date_from=date_from,
        train_date_to=date_to,
        params=params,
        dataset_hash=dataset_hash,
        reject_threshold=reject_threshold,
        data_provenance=data_provenance,
    )

    # Regression gates against champion
    regression: RegressionResult | None = None
    champion_macro_f1: float | None = None
    champion_path = find_latest_model(models_dir)

    if champion_path is not None:
        try:
            champ_model, champ_meta, champ_encoders = load_model_bundle(champion_path)
            champion_report = evaluate_model(
                champ_model, eval_df,
                cat_encoders=champ_encoders,
                holdout_users=holdout_users,
                reject_threshold=reject_threshold,
            )
            champion_macro_f1 = champion_report.macro_f1
            regression = check_regression_gates(champion_report, challenger_report, config)
        except (ValueError, OSError) as exc:
            logger.warning("Could not evaluate champion: %s — skipping regression gates", exc)

    # Promotion decision
    gates_passed = regression is None or regression.all_passed
    acceptance_passed = all(challenger_report.acceptance_checks.values())
    should_promote = gates_passed and acceptance_passed and not dry_run

    if should_promote:
        run_dir = save_model_bundle(
            model, metadata, metrics, cm_df, models_dir, cat_encoders=cat_encoders,
        )
        reason = "Promoted: all gates passed"
    else:
        run_dir = save_model_bundle(
            model, metadata, metrics, cm_df, out_dir / "rejected_models", cat_encoders=cat_encoders,
        )
        if dry_run:
            reason = "Dry run: model saved to rejected_models"
        elif not acceptance_passed:
            reason = "Rejected: acceptance checks failed"
        else:
            reason = "Rejected: regression gates failed"

    return RetrainResult(
        promoted=should_promote,
        champion_macro_f1=champion_macro_f1,
        challenger_macro_f1=challenger_report.macro_f1,
        regression=regression,
        dataset_snapshot=snapshot,
        run_dir=str(run_dir),
        reason=reason,
    )
