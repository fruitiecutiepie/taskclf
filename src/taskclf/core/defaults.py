"""Centralised default constants for taskclf.

Every project-wide magic number / string lives here.
Import these instead of hard-coding values in function signatures or CLI options.
"""

from __future__ import annotations

from typing import Final

# ── Timing / buckets ──
DEFAULT_BUCKET_SECONDS: Final[int] = 60
DEFAULT_POLL_SECONDS: Final[int] = 60
DEFAULT_IDLE_GAP_SECONDS: Final[float] = 300.0
DEFAULT_SMOOTH_WINDOW: Final[int] = 3
DEFAULT_APP_SWITCH_WINDOW_MINUTES: Final[int] = 5
DEFAULT_APP_SWITCH_WINDOW_15M: Final[int] = 15
DEFAULT_ROLLING_WINDOW_5: Final[int] = 5
DEFAULT_ROLLING_WINDOW_15: Final[int] = 15
DEFAULT_TITLE_HASH_BUCKETS: Final[int] = 256

# ── Paths ──
DEFAULT_OUT_DIR: Final[str] = "artifacts"
DEFAULT_DATA_DIR: Final[str] = "data/processed"
DEFAULT_RAW_AW_DIR: Final[str] = "data/raw/aw"
DEFAULT_MODELS_DIR: Final[str] = "models"

# ── ActivityWatch ──
DEFAULT_AW_HOST: Final[str] = "http://localhost:5600"
DEFAULT_AW_TIMEOUT_SECONDS: Final[int] = 10

# ── Privacy / hashing ──
DEFAULT_TITLE_SALT: Final[str] = "taskclf-default-salt"
DEFAULT_TITLE_POLICY: Final[str] = "hash_only"

# ── Aggregation ──
MIN_BLOCK_DURATION_SECONDS: Final[int] = 180

# ── Labeling ──
DEFAULT_LABEL_MAX_ASKS_PER_DAY: Final[int] = 20
DEFAULT_LABEL_CONFIDENCE_THRESHOLD: Final[float] = 0.55
DEFAULT_LABEL_SUMMARY_MINUTES: Final[int] = 30
DEFAULT_TRANSITION_MINUTES: Final[int] = 3

# ── Training ──
DEFAULT_NUM_BOOST_ROUND: Final[int] = 100
DEFAULT_REJECT_THRESHOLD: Final[float] = 0.55

# ── Personalization / calibration ──
DEFAULT_MIN_LABELED_WINDOWS: Final[int] = 200
DEFAULT_MIN_LABELED_DAYS: Final[int] = 3
DEFAULT_MIN_DISTINCT_LABELS: Final[int] = 3
DEFAULT_CALIBRATION_METHOD: Final[str] = "temperature"

# ── Baseline (rule-based) classifier ──
MIXED_UNKNOWN: Final[str] = "Mixed/Unknown"
BASELINE_IDLE_ACTIVE_THRESHOLD: Final[float] = 5.0
BASELINE_IDLE_RUN_THRESHOLD: Final[float] = 50.0
BASELINE_SCROLL_HIGH: Final[float] = 3.0
BASELINE_KEYS_LOW: Final[float] = 10.0
BASELINE_KEYS_HIGH: Final[float] = 30.0
BASELINE_SHORTCUT_HIGH: Final[float] = 1.0

# ── Drift / telemetry ──
DEFAULT_PSI_THRESHOLD: Final[float] = 0.2
DEFAULT_KS_ALPHA: Final[float] = 0.05
DEFAULT_REJECT_RATE_INCREASE_THRESHOLD: Final[float] = 0.10
DEFAULT_ENTROPY_SPIKE_MULTIPLIER: Final[float] = 2.0
DEFAULT_CLASS_SHIFT_THRESHOLD: Final[float] = 0.15
DEFAULT_DRIFT_REFERENCE_DAYS: Final[int] = 7
DEFAULT_DRIFT_WINDOW_DAYS: Final[int] = 7
DEFAULT_DRIFT_AUTO_LABEL_LIMIT: Final[int] = 50
DEFAULT_TELEMETRY_DIR: Final[str] = "artifacts/telemetry"
DEFAULT_PSI_BINS: Final[int] = 10

# ── Retraining ──
DEFAULT_RETRAIN_CADENCE_DAYS: Final[int] = 7
DEFAULT_CALIBRATOR_UPDATE_CADENCE_DAYS: Final[int] = 7
DEFAULT_DATA_LOOKBACK_DAYS: Final[int] = 30
DEFAULT_REGRESSION_TOLERANCE: Final[float] = 0.02

# ── Misc ──
DEFAULT_GIT_TIMEOUT_SECONDS: Final[int] = 5
DEFAULT_DUMMY_ROWS: Final[int] = 10
