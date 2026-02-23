"""Model bundle persistence: save, load, and metadata for trained model artifacts."""

from __future__ import annotations

import json
import random
import subprocess
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import lightgbm as lgb
import pandas as pd
from pydantic import BaseModel, Field

from taskclf.core.defaults import DEFAULT_GIT_TIMEOUT_SECONDS
from taskclf.core.schema import FeatureSchemaV1
from taskclf.core.types import LABEL_SET_V1


class ModelMetadata(BaseModel, frozen=True):
    """Immutable record stored alongside a trained model as ``metadata.json``.

    Captures the feature schema version/hash, label vocabulary, training
    date range, hyperparameters, and the git commit at training time so
    that inference can verify compatibility before predicting.
    """

    schema_version: str
    schema_hash: str
    label_set: list[str]
    train_date_from: str
    train_date_to: str
    params: dict[str, Any]
    git_commit: str
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


def _current_git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=DEFAULT_GIT_TIMEOUT_SECONDS,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def generate_run_id() -> str:
    """Produce a unique run directory name: ``YYYY-MM-DD_HHMMSS_run-XXXX``.

    Returns:
        A string like ``2026-02-19_013000_run-0042``.
    """
    now = datetime.now(UTC)
    suffix = f"{random.randint(0, 9999):04d}"
    return f"{now.strftime('%Y-%m-%d_%H%M%S')}_run-{suffix}"


def save_model_bundle(
    model: lgb.Booster,
    metadata: ModelMetadata,
    metrics: dict,
    confusion_df: pd.DataFrame,
    base_dir: Path,
    cat_encoders: dict | None = None,
) -> Path:
    """Persist a complete model bundle into ``base_dir/<run_id>/``.

    Writes the core files per the Model Bundle Contract plus an optional
    ``categorical_encoders.json`` mapping each categorical column to its
    sorted vocabulary list.

    Args:
        model: Trained LightGBM booster.
        metadata: Provenance record (schema hash, label set, params, etc.).
        metrics: Evaluation dict (as returned by
            :func:`~taskclf.core.metrics.compute_metrics`).
        confusion_df: Labelled confusion matrix for CSV export.
        base_dir: Parent directory (e.g. ``Path("models")``).
            A new ``<run_id>/`` subdirectory is created inside it.
        cat_encoders: Optional dict mapping categorical column names to
            fitted ``LabelEncoder`` instances.  Persisted as JSON
            vocabulary lists so inference can reconstruct them.

    Returns:
        Path to the newly created run directory.

    Raises:
        FileExistsError: If the generated run directory already exists.
    """
    run_id = generate_run_id()
    run_dir = base_dir / run_id
    if run_dir.exists():
        raise FileExistsError(f"Run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True)

    model.save_model(str(run_dir / "model.txt"))

    (run_dir / "metadata.json").write_text(
        json.dumps(metadata.model_dump(), indent=2)
    )

    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2)
    )

    confusion_df.to_csv(run_dir / "confusion_matrix.csv")

    if cat_encoders:
        vocab = {col: list(le.classes_) for col, le in cat_encoders.items()}
        (run_dir / "categorical_encoders.json").write_text(
            json.dumps(vocab, indent=2)
        )

    return run_dir


def load_model_bundle(
    run_dir: Path,
    *,
    validate_schema: bool = True,
    validate_labels: bool = True,
) -> tuple[lgb.Booster, ModelMetadata, dict | None]:
    """Load a model bundle and optionally validate schema hash and label set.

    Args:
        run_dir: Path to an existing run directory (e.g.
            ``models/2026-02-19_013000_run-0042/``).
        validate_schema: When ``True`` (the default), raise if the
            bundle's schema hash differs from the current
            ``FeatureSchemaV1.SCHEMA_HASH``.
        validate_labels: When ``True`` (the default), raise if the
            bundle's label set differs from the current ``LABEL_SET_V1``.

    Returns:
        A ``(model, metadata, cat_encoders)`` tuple where *cat_encoders*
        is a dict mapping column names to fitted ``LabelEncoder``
        instances, or ``None`` if no encoder file exists in the bundle.

    Raises:
        ValueError: If validation is enabled and the schema hash or label
            set recorded in the bundle does not match the running code.
    """
    from sklearn.preprocessing import LabelEncoder

    model = lgb.Booster(model_file=str(run_dir / "model.txt"))

    raw = json.loads((run_dir / "metadata.json").read_text())
    metadata = ModelMetadata.model_validate(raw)

    if validate_schema and metadata.schema_hash != FeatureSchemaV1.SCHEMA_HASH:
        raise ValueError(
            f"Schema hash mismatch: bundle has {metadata.schema_hash!r}, "
            f"current schema is {FeatureSchemaV1.SCHEMA_HASH!r}"
        )

    if validate_labels and sorted(metadata.label_set) != sorted(LABEL_SET_V1):
        raise ValueError(
            f"Label set mismatch: bundle has {sorted(metadata.label_set)!r}, "
            f"current label set is {sorted(LABEL_SET_V1)!r}"
        )

    cat_encoders: dict[str, LabelEncoder] | None = None
    enc_path = run_dir / "categorical_encoders.json"
    if enc_path.exists():
        vocab = json.loads(enc_path.read_text())
        cat_encoders = {}
        for col, classes in vocab.items():
            le = LabelEncoder()
            le.classes_ = __import__("numpy").array(classes)
            cat_encoders[col] = le

    return model, metadata, cat_encoders


def build_metadata(
    label_set: list[str],
    train_date_from: date,
    train_date_to: date,
    params: dict[str, Any],
) -> ModelMetadata:
    """Convenience builder that fills in schema info and git commit.

    Args:
        label_set: Task-type labels used during training.
        train_date_from: First date of the training range.
        train_date_to: Last date (inclusive) of the training range.
        params: LightGBM (or other model) hyperparameters dict.

    Returns:
        A populated ``ModelMetadata`` instance.
    """
    return ModelMetadata(
        schema_version=FeatureSchemaV1.VERSION,
        schema_hash=FeatureSchemaV1.SCHEMA_HASH,
        label_set=sorted(label_set),
        train_date_from=train_date_from.isoformat(),
        train_date_to=train_date_to.isoformat(),
        params=params,
        git_commit=_current_git_commit(),
    )
