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

# ── Training ──
DEFAULT_NUM_BOOST_ROUND: Final[int] = 100
DEFAULT_TRAIN_SPLIT_RATIO: Final[float] = 0.8

# ── Misc ──
DEFAULT_GIT_TIMEOUT_SECONDS: Final[int] = 5
DEFAULT_DUMMY_ROWS: Final[int] = 10
