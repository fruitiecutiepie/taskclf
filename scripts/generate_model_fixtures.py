"""Generate test fixture model bundles under tests/fixtures/models/.

Run from the repo root:

    uv run python scripts/generate_model_fixtures.py

Creates sample bundles:
  - best_bundle:            compatible, highest macro_f1 (0.88) — selection winner
  - good_bundle:            compatible, mid macro_f1 (0.82) — ranking middle
  - second_good_bundle:     compatible, lower macro_f1 (0.75) — ranking bottom
  - bad_schema_bundle:      wrong schema_hash, for testing rejection paths
  - missing_metrics_bundle: has metadata.json only, no metrics.json
  - corrupt_json_bundle:    has both files but metrics.json is not valid JSON
"""

from __future__ import annotations

import json
from pathlib import Path

from taskclf.core.model_io import ModelMetadata
from taskclf.core.schema import FeatureSchemaV1
from taskclf.core.types import LABEL_SET_V1

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "models"

BUNDLES: dict[str, dict] = {
    "best_bundle": {
        "metadata": ModelMetadata(
            schema_version=FeatureSchemaV1.VERSION,
            schema_hash=FeatureSchemaV1.SCHEMA_HASH,
            label_set=sorted(LABEL_SET_V1),
            train_date_from="2026-01-15",
            train_date_to="2026-02-10",
            params={"num_leaves": 63, "learning_rate": 0.03, "n_estimators": 400},
            git_commit="fff000aaa111",
            dataset_hash="cccc777788889999",
            reject_threshold=0.30,
            data_provenance="real",
            created_at="2026-02-10T00:00:00+00:00",
        ),
        "metrics": {
            "macro_f1": 0.88,
            "weighted_f1": 0.91,
            "confusion_matrix": [
                [48, 1, 0, 0, 0, 0, 0, 1],
                [1, 44, 1, 1, 0, 0, 0, 3],
                [0, 2, 42, 1, 1, 0, 0, 4],
                [0, 0, 1, 46, 0, 1, 0, 2],
                [0, 0, 0, 1, 38, 1, 0, 0],
                [0, 0, 0, 0, 1, 44, 0, 5],
                [0, 0, 0, 0, 0, 0, 49, 1],
                [0, 1, 0, 0, 0, 1, 0, 48],
            ],
            "label_names": sorted(LABEL_SET_V1),
        },
    },
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
    "second_good_bundle": {
        "metadata": ModelMetadata(
            schema_version=FeatureSchemaV1.VERSION,
            schema_hash=FeatureSchemaV1.SCHEMA_HASH,
            label_set=sorted(LABEL_SET_V1),
            train_date_from="2025-12-01",
            train_date_to="2025-12-31",
            params={"num_leaves": 31, "learning_rate": 0.05, "n_estimators": 150},
            git_commit="bbb222ccc333",
            dataset_hash="dddd888899990000",
            reject_threshold=0.40,
            data_provenance="real",
            created_at="2026-01-10T00:00:00+00:00",
        ),
        "metrics": {
            "macro_f1": 0.75,
            "weighted_f1": 0.78,
            "confusion_matrix": [
                [40, 3, 2, 1, 0, 2, 0, 2],
                [4, 35, 3, 2, 1, 1, 0, 4],
                [2, 4, 32, 3, 2, 1, 1, 5],
                [1, 2, 3, 38, 2, 1, 1, 2],
                [1, 1, 2, 2, 30, 3, 1, 0],
                [2, 1, 1, 2, 3, 36, 2, 3],
                [0, 1, 0, 1, 0, 2, 44, 2],
                [2, 3, 2, 1, 1, 3, 1, 37],
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


EDGE_CASE_BUNDLES: dict[str, dict] = {
    "missing_metrics_bundle": {
        "metadata": ModelMetadata(
            schema_version=FeatureSchemaV1.VERSION,
            schema_hash=FeatureSchemaV1.SCHEMA_HASH,
            label_set=sorted(LABEL_SET_V1),
            train_date_from="2026-01-01",
            train_date_to="2026-01-15",
            params={"num_leaves": 15},
            git_commit="def456abc789",
            dataset_hash="aaaa111122223333",
            data_provenance="real",
            created_at="2026-01-20T00:00:00+00:00",
        ),
        "metrics": None,
    },
    "corrupt_json_bundle": {
        "metadata": ModelMetadata(
            schema_version=FeatureSchemaV1.VERSION,
            schema_hash=FeatureSchemaV1.SCHEMA_HASH,
            label_set=sorted(LABEL_SET_V1),
            train_date_from="2026-01-01",
            train_date_to="2026-01-15",
            params={"num_leaves": 15},
            git_commit="789abc123def",
            dataset_hash="bbbb444455556666",
            data_provenance="synthetic",
            created_at="2026-01-25T00:00:00+00:00",
        ),
        "metrics_raw": "{not valid json at all!!!",
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

    for name, bundle in EDGE_CASE_BUNDLES.items():
        d = FIXTURES_DIR / name
        d.mkdir(parents=True, exist_ok=True)

        (d / "metadata.json").write_text(
            json.dumps(bundle["metadata"].model_dump(), indent=2) + "\n"
        )

        if bundle.get("metrics") is not None:
            (d / "metrics.json").write_text(
                json.dumps(bundle["metrics"], indent=2) + "\n"
            )
        elif bundle.get("metrics_raw") is not None:
            (d / "metrics.json").write_text(bundle["metrics_raw"] + "\n")

        print(f"wrote {d}")


if __name__ == "__main__":
    main()
