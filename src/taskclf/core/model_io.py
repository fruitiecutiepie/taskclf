"""Model bundle persistence: save, load, and metadata for trained model artifacts."""

from __future__ import annotations

import json
import random
import subprocess
from datetime import date, datetime
from pathlib import Path
from typing import Any

import lightgbm as lgb
import pandas as pd
from pydantic import BaseModel, Field

from taskclf.core.schema import FeatureSchemaV1


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
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


def _current_git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def generate_run_id() -> str:
    """Produce a unique run directory name: ``YYYY-MM-DD_HHMMSS_run-XXXX``."""
    now = datetime.utcnow()
    suffix = f"{random.randint(0, 9999):04d}"
    return f"{now.strftime('%Y-%m-%d_%H%M%S')}_run-{suffix}"


def save_model_bundle(
    model: lgb.Booster,
    metadata: ModelMetadata,
    metrics: dict,
    confusion_df: pd.DataFrame,
    base_dir: Path,
) -> Path:
    """Persist a complete model bundle into ``base_dir/<run_id>/``.

    Writes four files per the Model Bundle Contract:
    ``model.txt``, ``metadata.json``, ``metrics.json``, ``confusion_matrix.csv``.

    Returns the path to the run directory.
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

    return run_dir


def load_model_bundle(
    run_dir: Path,
    *,
    validate_schema: bool = True,
) -> tuple[lgb.Booster, ModelMetadata]:
    """Load a model bundle and optionally validate the schema hash.

    Raises ``ValueError`` if *validate_schema* is True and the bundle's
    schema hash does not match the current ``FeatureSchemaV1.SCHEMA_HASH``.
    """
    model = lgb.Booster(model_file=str(run_dir / "model.txt"))

    raw = json.loads((run_dir / "metadata.json").read_text())
    metadata = ModelMetadata.model_validate(raw)

    if validate_schema and metadata.schema_hash != FeatureSchemaV1.SCHEMA_HASH:
        raise ValueError(
            f"Schema hash mismatch: bundle has {metadata.schema_hash!r}, "
            f"current schema is {FeatureSchemaV1.SCHEMA_HASH!r}"
        )

    return model, metadata


def build_metadata(
    label_set: list[str],
    train_date_from: date,
    train_date_to: date,
    params: dict[str, Any],
) -> ModelMetadata:
    """Convenience builder that fills in schema info and git commit."""
    return ModelMetadata(
        schema_version=FeatureSchemaV1.VERSION,
        schema_hash=FeatureSchemaV1.SCHEMA_HASH,
        label_set=sorted(label_set),
        train_date_from=train_date_from.isoformat(),
        train_date_to=train_date_to.isoformat(),
        params=params,
        git_commit=_current_git_commit(),
    )
