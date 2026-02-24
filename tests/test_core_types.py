"""Tests for core data contracts: FeatureRow privacy/validation and LabelSpan invariants.

Covers: TC-CORE-001 through TC-CORE-004, TC-LABEL-004.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

import pytest
from pydantic import ValidationError

from taskclf.core.types import CoreLabel, FeatureRow, LabelSpan


class TestFeatureRowValidation:
    """TC-CORE-001 / TC-CORE-002: required-field and extra-field checks."""

    def test_valid_construction(self, valid_feature_row_data: dict[str, Any]) -> None:
        row = FeatureRow.model_validate(valid_feature_row_data)
        assert row.app_id == "com.apple.Terminal"

    @pytest.mark.parametrize(
        "missing_field",
        [
            "bucket_start_ts",
            "schema_version",
            "schema_hash",
            "source_ids",
            "app_id",
            "window_title_hash",
            "is_browser",
            "hour_of_day",
        ],
    )
    def test_rejects_missing_required_field(
        self, valid_feature_row_data: dict[str, Any], missing_field: str
    ) -> None:
        data = {k: v for k, v in valid_feature_row_data.items() if k != missing_field}
        with pytest.raises(ValidationError):
            FeatureRow.model_validate(data)

    def test_nullable_fields_default_to_none(
        self, valid_feature_row_data: dict[str, Any]
    ) -> None:
        data = {
            k: v
            for k, v in valid_feature_row_data.items()
            if k not in ("keys_per_min", "clicks_per_min", "mouse_distance")
        }
        row = FeatureRow.model_validate(data)
        assert row.keys_per_min is None
        assert row.clicks_per_min is None
        assert row.mouse_distance is None

    def test_source_ids_must_be_nonempty(
        self, valid_feature_row_data: dict[str, Any]
    ) -> None:
        data = {**valid_feature_row_data, "source_ids": []}
        with pytest.raises(ValidationError):
            FeatureRow.model_validate(data)

    def test_hour_of_day_range(
        self, valid_feature_row_data: dict[str, Any]
    ) -> None:
        with pytest.raises(ValidationError):
            FeatureRow.model_validate({**valid_feature_row_data, "hour_of_day": 24})
        with pytest.raises(ValidationError):
            FeatureRow.model_validate({**valid_feature_row_data, "hour_of_day": -1})

    def test_day_of_week_range(
        self, valid_feature_row_data: dict[str, Any]
    ) -> None:
        with pytest.raises(ValidationError):
            FeatureRow.model_validate({**valid_feature_row_data, "day_of_week": 7})


class TestFeatureRowPrivacy:
    """TC-CORE-003 / TC-CORE-004: prohibited raw_* fields are rejected."""

    def test_rejects_raw_keystrokes(
        self, valid_feature_row_data: dict[str, Any]
    ) -> None:
        data = {**valid_feature_row_data, "raw_keystrokes": "secret typing"}
        with pytest.raises(ValidationError, match="raw_keystrokes"):
            FeatureRow.model_validate(data)

    def test_rejects_raw_window_title(
        self, valid_feature_row_data: dict[str, Any]
    ) -> None:
        data = {**valid_feature_row_data, "raw_window_title": "My Secret Doc"}
        with pytest.raises(ValidationError, match="raw_window_title"):
            FeatureRow.model_validate(data)

    def test_rejects_any_raw_prefix_field(
        self, valid_feature_row_data: dict[str, Any]
    ) -> None:
        data = {**valid_feature_row_data, "raw_clipboard": "paste data"}
        with pytest.raises(ValidationError, match="raw_clipboard"):
            FeatureRow.model_validate(data)


class TestLabelSpanValidation:
    """TC-LABEL-004 and label-set enforcement."""

    def test_valid_construction(self) -> None:
        span = LabelSpan(
            start_ts=dt.datetime(2025, 6, 15, 10, 0),
            end_ts=dt.datetime(2025, 6, 15, 10, 5),
            label="Build",
            provenance="manual",
        )
        assert span.label == "Build"

    def test_rejects_end_before_start(self) -> None:
        with pytest.raises(ValidationError, match="end_ts"):
            LabelSpan(
                start_ts=dt.datetime(2025, 6, 15, 10, 5),
                end_ts=dt.datetime(2025, 6, 15, 10, 0),
                label="Build",
                provenance="manual",
            )

    def test_rejects_end_equal_to_start(self) -> None:
        ts = dt.datetime(2025, 6, 15, 10, 0)
        with pytest.raises(ValidationError, match="end_ts"):
            LabelSpan(start_ts=ts, end_ts=ts, label="Build", provenance="manual")

    def test_rejects_unknown_label(self) -> None:
        with pytest.raises(ValidationError, match="Unknown label"):
            LabelSpan(
                start_ts=dt.datetime(2025, 6, 15, 10, 0),
                end_ts=dt.datetime(2025, 6, 15, 10, 5),
                label="playing_games",
                provenance="manual",
            )


class TestCoreLabelMatchesSchema:
    """Ensure CoreLabel enum stays in sync with schema/labels_v1.json."""

    def test_labels_match_json_schema(self) -> None:
        import json
        from pathlib import Path

        schema_path = Path(__file__).resolve().parents[1] / "schema" / "labels_v1.json"
        schema = json.loads(schema_path.read_text())

        json_labels = [entry["name"] for entry in schema["labels"]]
        enum_labels = [member.value for member in CoreLabel]

        assert enum_labels == json_labels

    def test_label_ids_match_enum_order(self) -> None:
        import json
        from pathlib import Path

        schema_path = Path(__file__).resolve().parents[1] / "schema" / "labels_v1.json"
        schema = json.loads(schema_path.read_text())

        for idx, member in enumerate(CoreLabel):
            assert schema["labels"][idx]["id"] == idx
            assert schema["labels"][idx]["name"] == member.value

    def test_num_classes_matches(self) -> None:
        import json
        from pathlib import Path

        schema_path = Path(__file__).resolve().parents[1] / "schema" / "labels_v1.json"
        schema = json.loads(schema_path.read_text())

        assert schema["num_classes"] == len(CoreLabel)
