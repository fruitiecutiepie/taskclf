"""Generate test fixture model bundles under tests/fixtures/models/.

Run from the repo root:

    uv run python scripts/generate_model_fixtures.py

Creates two sample bundles:
  - good_bundle:       valid schema_hash, passes load_model_bundle() checks
  - bad_schema_bundle: wrong schema_hash, for testing rejection paths
"""

from __future__ import annotations

import json
from pathlib import Path

from taskclf.core.model_io import ModelMetadata
from taskclf.core.schema import FeatureSchemaV1
from taskclf.core.types import LABEL_SET_V1

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "models"

BUNDLES: dict[str, dict] = {
    "good_bundle": {
        "metadata": ModelMetadata(
            schema_version=FeatureSchemaV1.VERSION,
            schema_hash=FeatureSchemaV1.SCHEMA_HASH,
            label_set=sorted(LABEL_SET_V1),
            train_date_from="2026-01-01",
            train_date_to="2026-01-31",
            params={"num_leaves": 31, "learning_rate": 0.05, "n_estimators": 200},
            git_commit="abc123def456",
            dataset_hash="a1b2c3d4e5f6g7h8",
            reject_threshold=0.35,
            data_provenance="real",
            created_at="2026-02-01T00:00:00+00:00",
        ),
        "metrics": {
            "macro_f1": 0.82,
            "weighted_f1": 0.85,
            "confusion_matrix": [
                [45, 2, 1, 0, 0, 1, 0, 1],
                [3, 40, 2, 1, 0, 0, 0, 4],
                [1, 3, 38, 2, 1, 0, 0, 5],
                [0, 1, 2, 42, 1, 1, 0, 3],
                [0, 0, 1, 1, 35, 2, 0, 1],
                [1, 0, 0, 1, 2, 40, 1, 5],
                [0, 0, 0, 0, 0, 1, 48, 1],
                [1, 2, 1, 0, 0, 2, 0, 44],
            ],
            "label_names": sorted(LABEL_SET_V1),
        },
    },
    "bad_schema_bundle": {
        "metadata": ModelMetadata(
            schema_version="v1",
            schema_hash="deadbeef_wrong_hash",
            label_set=sorted(LABEL_SET_V1),
            train_date_from="2026-01-01",
            train_date_to="2026-01-31",
            params={"num_leaves": 31},
            git_commit="unknown",
            dataset_hash="0000000000000000",
            data_provenance="synthetic",
            created_at="2026-01-15T00:00:00+00:00",
        ),
        "metrics": {
            "macro_f1": 0.40,
            "weighted_f1": 0.45,
            "confusion_matrix": [
                [20, 5, 5, 3, 2, 5, 2, 8],
                [6, 18, 4, 4, 3, 3, 2, 10],
                [5, 4, 15, 6, 3, 4, 3, 10],
                [3, 4, 6, 20, 4, 3, 2, 8],
                [2, 3, 3, 4, 18, 5, 3, 12],
                [5, 3, 4, 3, 5, 22, 3, 5],
                [2, 2, 3, 2, 3, 3, 30, 5],
                [4, 5, 4, 3, 3, 4, 2, 25],
            ],
            "label_names": sorted(LABEL_SET_V1),
        },
    },
}


def main() -> None:
    for name, bundle in BUNDLES.items():
        d = FIXTURES_DIR / name
        d.mkdir(parents=True, exist_ok=True)

        (d / "metadata.json").write_text(
            json.dumps(bundle["metadata"].model_dump(), indent=2) + "\n"
        )
        (d / "metrics.json").write_text(
            json.dumps(bundle["metrics"], indent=2) + "\n"
        )
        print(f"wrote {d}")


if __name__ == "__main__":
    main()
