"""Smoke tests for core.defaults: every public constant is importable and correctly typed.

Catches accidental deletions or type regressions.
"""

from __future__ import annotations

from pathlib import Path

from taskclf.core import defaults


class TestDefaultsTypes:
    """All public constants have the expected Python type."""

    def test_int_constants(self) -> None:
        int_names = [
            "DEFAULT_BUCKET_SECONDS",
            "DEFAULT_POLL_SECONDS",
            "DEFAULT_SMOOTH_WINDOW",
            "DEFAULT_APP_SWITCH_WINDOW_MINUTES",
            "DEFAULT_APP_SWITCH_WINDOW_15M",
            "DEFAULT_ROLLING_WINDOW_5",
            "DEFAULT_ROLLING_WINDOW_15",
            "DEFAULT_TITLE_HASH_BUCKETS",
            "DEFAULT_AW_TIMEOUT_SECONDS",
            "MIN_BLOCK_DURATION_SECONDS",
            "DEFAULT_LABEL_MAX_ASKS_PER_DAY",
            "DEFAULT_LABEL_SUMMARY_MINUTES",
            "DEFAULT_TRANSITION_MINUTES",
            "DEFAULT_NUM_BOOST_ROUND",
            "DEFAULT_MIN_LABELED_WINDOWS",
            "DEFAULT_MIN_LABELED_DAYS",
            "DEFAULT_MIN_DISTINCT_LABELS",
            "DEFAULT_DRIFT_REFERENCE_DAYS",
            "DEFAULT_DRIFT_WINDOW_DAYS",
            "DEFAULT_DRIFT_AUTO_LABEL_LIMIT",
            "DEFAULT_PSI_BINS",
            "DEFAULT_RETRAIN_CADENCE_DAYS",
            "DEFAULT_CALIBRATOR_UPDATE_CADENCE_DAYS",
            "DEFAULT_DATA_LOOKBACK_DAYS",
            "DEFAULT_GIT_TIMEOUT_SECONDS",
            "DEFAULT_DUMMY_ROWS",
        ]
        for name in int_names:
            value = getattr(defaults, name)
            assert isinstance(value, int), f"{name} should be int, got {type(value)}"

    def test_float_constants(self) -> None:
        float_names = [
            "DEFAULT_IDLE_GAP_SECONDS",
            "DEFAULT_LABEL_CONFIDENCE_THRESHOLD",
            "DEFAULT_REJECT_THRESHOLD",
            "BASELINE_IDLE_ACTIVE_THRESHOLD",
            "BASELINE_IDLE_RUN_THRESHOLD",
            "BASELINE_SCROLL_HIGH",
            "BASELINE_KEYS_LOW",
            "BASELINE_KEYS_HIGH",
            "BASELINE_SHORTCUT_HIGH",
            "DEFAULT_PSI_THRESHOLD",
            "DEFAULT_KS_ALPHA",
            "DEFAULT_REJECT_RATE_INCREASE_THRESHOLD",
            "DEFAULT_ENTROPY_SPIKE_MULTIPLIER",
            "DEFAULT_CLASS_SHIFT_THRESHOLD",
            "DEFAULT_REGRESSION_TOLERANCE",
        ]
        for name in float_names:
            value = getattr(defaults, name)
            assert isinstance(value, float), (
                f"{name} should be float, got {type(value)}"
            )

    def test_str_constants(self) -> None:
        str_names = [
            "DEFAULT_OUT_DIR",
            "DEFAULT_DATA_DIR",
            "DEFAULT_RAW_AW_DIR",
            "DEFAULT_MODELS_DIR",
            "DEFAULT_AW_HOST",
            "DEFAULT_TITLE_SALT",
            "DEFAULT_TITLE_POLICY",
            "DEFAULT_CALIBRATION_METHOD",
            "MIXED_UNKNOWN",
            "DEFAULT_TELEMETRY_DIR",
            "DEFAULT_LOG_DIR",
        ]
        for name in str_names:
            value = getattr(defaults, name)
            assert isinstance(value, str), f"{name} should be str, got {type(value)}"

    def test_path_constants_are_absolute(self) -> None:
        path_names = [
            "DEFAULT_OUT_DIR",
            "DEFAULT_DATA_DIR",
            "DEFAULT_RAW_AW_DIR",
            "DEFAULT_MODELS_DIR",
            "DEFAULT_TELEMETRY_DIR",
            "DEFAULT_LOG_DIR",
        ]
        for name in path_names:
            value = getattr(defaults, name)
            assert Path(value).is_absolute(), (
                f"{name} should be absolute, got {value!r}"
            )

    def test_all_public_names_covered(self) -> None:
        """Every non-dunder, non-import name in the module is checked above."""
        _imports = {"Final", "annotations", "taskclf_home"}
        public = {
            n for n in dir(defaults) if not n.startswith("_") and n not in _imports
        }
        covered = {
            "DEFAULT_BUCKET_SECONDS",
            "DEFAULT_POLL_SECONDS",
            "DEFAULT_IDLE_GAP_SECONDS",
            "DEFAULT_SMOOTH_WINDOW",
            "DEFAULT_APP_SWITCH_WINDOW_MINUTES",
            "DEFAULT_APP_SWITCH_WINDOW_15M",
            "DEFAULT_ROLLING_WINDOW_5",
            "DEFAULT_ROLLING_WINDOW_15",
            "DEFAULT_TITLE_HASH_BUCKETS",
            "DEFAULT_OUT_DIR",
            "DEFAULT_DATA_DIR",
            "DEFAULT_RAW_AW_DIR",
            "DEFAULT_MODELS_DIR",
            "DEFAULT_AW_HOST",
            "DEFAULT_AW_TIMEOUT_SECONDS",
            "DEFAULT_TITLE_SALT",
            "DEFAULT_TITLE_POLICY",
            "MIN_BLOCK_DURATION_SECONDS",
            "DEFAULT_LABEL_MAX_ASKS_PER_DAY",
            "DEFAULT_LABEL_CONFIDENCE_THRESHOLD",
            "DEFAULT_LABEL_SUMMARY_MINUTES",
            "DEFAULT_TRANSITION_MINUTES",
            "DEFAULT_NUM_BOOST_ROUND",
            "DEFAULT_REJECT_THRESHOLD",
            "DEFAULT_MIN_LABELED_WINDOWS",
            "DEFAULT_MIN_LABELED_DAYS",
            "DEFAULT_MIN_DISTINCT_LABELS",
            "DEFAULT_CALIBRATION_METHOD",
            "MIXED_UNKNOWN",
            "BASELINE_IDLE_ACTIVE_THRESHOLD",
            "BASELINE_IDLE_RUN_THRESHOLD",
            "BASELINE_SCROLL_HIGH",
            "BASELINE_KEYS_LOW",
            "BASELINE_KEYS_HIGH",
            "BASELINE_SHORTCUT_HIGH",
            "DEFAULT_PSI_THRESHOLD",
            "DEFAULT_KS_ALPHA",
            "DEFAULT_REJECT_RATE_INCREASE_THRESHOLD",
            "DEFAULT_ENTROPY_SPIKE_MULTIPLIER",
            "DEFAULT_CLASS_SHIFT_THRESHOLD",
            "DEFAULT_DRIFT_REFERENCE_DAYS",
            "DEFAULT_DRIFT_WINDOW_DAYS",
            "DEFAULT_DRIFT_AUTO_LABEL_LIMIT",
            "DEFAULT_TELEMETRY_DIR",
            "DEFAULT_PSI_BINS",
            "DEFAULT_RETRAIN_CADENCE_DAYS",
            "DEFAULT_CALIBRATOR_UPDATE_CADENCE_DAYS",
            "DEFAULT_DATA_LOOKBACK_DAYS",
            "DEFAULT_REGRESSION_TOLERANCE",
            "DEFAULT_GIT_TIMEOUT_SECONDS",
            "DEFAULT_DUMMY_ROWS",
            "DEFAULT_LOG_DIR",
        }
        missing = public - covered
        assert not missing, f"Uncovered public names: {missing}"
