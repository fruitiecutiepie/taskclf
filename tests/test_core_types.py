"""Tests for core data contracts: FeatureRow privacy/validation, LabelSpan invariants,
and Event protocol conformance.

Covers: TC-CORE-001 through TC-CORE-005, TC-LABEL-004,
TC-TYPES-001..002 (Event protocol), TC-TYPES-003..005 (LabelSpan confidence NaN),
TC-TYPES-006..007 (LabelSpan extend_forward).
"""

from __future__ import annotations

import datetime as dt
from typing import Any

import pytest
from pydantic import ValidationError

from taskclf.core.types import (
    AxisDecision,
    CoreLabel,
    Event,
    FeatureRow,
    LabelEnvelope,
    LabelSpan,
    Mode,
    SemanticLabel,
    SupportState,
    TitlePolicy,
)


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

    def test_hour_of_day_range(self, valid_feature_row_data: dict[str, Any]) -> None:
        with pytest.raises(ValidationError):
            FeatureRow.model_validate({**valid_feature_row_data, "hour_of_day": 24})
        with pytest.raises(ValidationError):
            FeatureRow.model_validate({**valid_feature_row_data, "hour_of_day": -1})

    def test_day_of_week_range(self, valid_feature_row_data: dict[str, Any]) -> None:
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


class TestTitlePolicyGating:
    """TC-CORE-005: raw_window_title is allowed only with RAW_WINDOW_TITLE_OPT_IN context."""

    def test_rejects_raw_title_no_context(
        self, valid_feature_row_data: dict[str, Any]
    ) -> None:
        data = {**valid_feature_row_data, "raw_window_title": "Secret Doc"}
        with pytest.raises(ValidationError, match="raw_window_title"):
            FeatureRow.model_validate(data)

    def test_rejects_raw_title_hash_only(
        self, valid_feature_row_data: dict[str, Any]
    ) -> None:
        data = {**valid_feature_row_data, "raw_window_title": "Secret Doc"}
        ctx = {"title_policy": TitlePolicy.HASH_ONLY}
        with pytest.raises(ValidationError, match="raw_window_title"):
            FeatureRow.model_validate(data, context=ctx)

    def test_accepts_raw_title_raw_window_title_opt_in(
        self, valid_feature_row_data: dict[str, Any]
    ) -> None:
        data = {**valid_feature_row_data, "raw_window_title": "My Document.docx"}
        ctx = {"title_policy": TitlePolicy.RAW_WINDOW_TITLE_OPT_IN}
        row = FeatureRow.model_validate(data, context=ctx)
        assert row.raw_window_title == "My Document.docx"

    def test_still_rejects_other_raw_fields_with_opt_in(
        self, valid_feature_row_data: dict[str, Any]
    ) -> None:
        data = {**valid_feature_row_data, "raw_keystrokes": "secret typing"}
        ctx = {"title_policy": TitlePolicy.RAW_WINDOW_TITLE_OPT_IN}
        with pytest.raises(ValidationError, match="raw_keystrokes"):
            FeatureRow.model_validate(data, context=ctx)

    def test_model_dump_excludes_raw_title(
        self, valid_feature_row_data: dict[str, Any]
    ) -> None:
        data = {**valid_feature_row_data, "raw_window_title": "My Document.docx"}
        ctx = {"title_policy": TitlePolicy.RAW_WINDOW_TITLE_OPT_IN}
        row = FeatureRow.model_validate(data, context=ctx)
        dumped = row.model_dump()
        assert "raw_window_title" not in dumped

    def test_raw_title_defaults_to_none(
        self, valid_feature_row_data: dict[str, Any]
    ) -> None:
        row = FeatureRow.model_validate(valid_feature_row_data)
        assert row.raw_window_title is None


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


class TestLabelEnvelopeCompatibility:
    """Legacy payloads should lift `label` into the new semantic field."""

    def test_legacy_label_field_lifts_to_semantic(self) -> None:
        envelope = LabelEnvelope.model_validate(
            {
                "rule_version": "2026-04-25.1",
                "generated_at": "2026-04-25T00:00:00+00:00",
                "evidence_window_ms": 30000,
                "inference_window_ms": 180000,
                "label": {
                    "mode": {"value": "Produce", "confidence": 0.9},
                    "support_state": "Supported",
                },
            }
        )

        assert isinstance(envelope.semantic, SemanticLabel)
        assert envelope.semantic.mode == AxisDecision[Mode](
            value=Mode.Produce, confidence=0.9
        )
        assert envelope.semantic.support_state == SupportState.Supported
        assert envelope.label is envelope.semantic

        dumped = envelope.model_dump(mode="json", exclude_none=True)
        assert "label" not in dumped
        assert dumped["semantic"]["mode"]["value"] == "Produce"


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


# ---------------------------------------------------------------------------
# Event protocol
# ---------------------------------------------------------------------------


class TestEventProtocol:
    """TC-TYPES-001..002: runtime_checkable Event protocol."""

    def test_conforming_object_satisfies_protocol(self) -> None:
        class _FakeEvent:
            @property
            def timestamp(self) -> dt.datetime:
                return dt.datetime(2025, 6, 15, 10, 0)

            @property
            def duration_seconds(self) -> float:
                return 60.0

            @property
            def app_id(self) -> str:
                return "com.example.App"

            @property
            def window_title_hash(self) -> str:
                return "abc123"

            @property
            def is_browser(self) -> bool:
                return False

            @property
            def is_editor(self) -> bool:
                return True

            @property
            def is_terminal(self) -> bool:
                return False

            @property
            def app_category(self) -> str:
                return "editor"

        assert isinstance(_FakeEvent(), Event)

    def test_non_conforming_object_fails_protocol(self) -> None:
        class _Incomplete:
            @property
            def timestamp(self) -> dt.datetime:
                return dt.datetime(2025, 6, 15, 10, 0)

        assert not isinstance(_Incomplete(), Event)


# ---------------------------------------------------------------------------
# LabelSpan.confidence NaN coercion
# ---------------------------------------------------------------------------


class TestLabelSpanConfidenceNaN:
    """TC-TYPES-003..005: NaN confidence is coerced to None."""

    def test_nan_confidence_becomes_none(self) -> None:
        span = LabelSpan(
            start_ts=dt.datetime(2025, 6, 15, 10, 0),
            end_ts=dt.datetime(2025, 6, 15, 10, 5),
            label="Build",
            provenance="manual",
            confidence=float("nan"),
        )
        assert span.confidence is None

    def test_valid_confidence_preserved(self) -> None:
        span = LabelSpan(
            start_ts=dt.datetime(2025, 6, 15, 10, 0),
            end_ts=dt.datetime(2025, 6, 15, 10, 5),
            label="Build",
            provenance="manual",
            confidence=0.8,
        )
        assert span.confidence == 0.8

    def test_none_confidence_preserved(self) -> None:
        span = LabelSpan(
            start_ts=dt.datetime(2025, 6, 15, 10, 0),
            end_ts=dt.datetime(2025, 6, 15, 10, 5),
            label="Build",
            provenance="manual",
            confidence=None,
        )
        assert span.confidence is None


# ---------------------------------------------------------------------------
# LabelSpan.extend_forward
# ---------------------------------------------------------------------------


class TestLabelSpanExtendForward:
    """TC-TYPES-006..007: extend_forward field."""

    def test_defaults_to_false(self) -> None:
        span = LabelSpan(
            start_ts=dt.datetime(2025, 6, 15, 10, 0),
            end_ts=dt.datetime(2025, 6, 15, 10, 5),
            label="Build",
            provenance="manual",
        )
        assert span.extend_forward is False

    def test_explicit_true(self) -> None:
        span = LabelSpan(
            start_ts=dt.datetime(2025, 6, 15, 10, 0),
            end_ts=dt.datetime(2025, 6, 15, 10, 5),
            label="Build",
            provenance="manual",
            extend_forward=True,
        )
        assert span.extend_forward is True
