# TODO: real event→minute features pipeline (currently dummy data)

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd

from taskclf.core.hashing import stable_hash
from taskclf.core.schema import FeatureSchemaV1
from taskclf.core.store import write_parquet
from taskclf.core.types import FeatureRow

_DUMMY_APPS: list[tuple[str, bool, bool, bool]] = [
    # (app_id, is_browser, is_editor, is_terminal)
    ("com.apple.Terminal", False, False, True),
    ("org.mozilla.firefox", True, False, False),
    ("com.microsoft.VSCode", False, True, False),
    ("com.apple.mail", False, False, False),
    ("us.zoom.xos", False, False, False),
    ("com.tinyspeck.slackmacgap", False, False, False),
    ("com.google.Chrome", True, False, False),
    ("com.jetbrains.intellij", False, True, False),
    ("com.apple.finder", False, False, False),
    ("com.apple.Notes", False, False, False),
]


def generate_dummy_features(
    date: dt.date, n_rows: int = 10
) -> list[FeatureRow]:
    """Create *n_rows* synthetic FeatureRow instances spanning *date*.

    Args:
        date: The calendar date to generate buckets for (hours 9-17).
        n_rows: Number of rows to generate.

    Returns:
        Validated ``FeatureRow`` instances with dummy app/keyboard/mouse data.
    """
    rows: list[FeatureRow] = []
    day_of_week = date.weekday()

    for i in range(n_rows):
        hour = 9 + (i * 8 // max(n_rows, 1))
        minute = (i * 7) % 60
        ts = dt.datetime(date.year, date.month, date.day, hour, minute)

        app_id, is_browser, is_editor, is_terminal = _DUMMY_APPS[
            i % len(_DUMMY_APPS)
        ]
        title_hash = stable_hash(f"window-title-{app_id}-{i}")

        rows.append(
            FeatureRow(
                bucket_start_ts=ts,
                schema_version=FeatureSchemaV1.VERSION,
                schema_hash=FeatureSchemaV1.SCHEMA_HASH,
                source_ids=[f"dummy-{i:03d}"],
                app_id=app_id,
                window_title_hash=title_hash,
                is_browser=is_browser,
                is_editor=is_editor,
                is_terminal=is_terminal,
                app_switch_count_last_5m=i % 5,
                keys_per_min=float(40 + i * 10),
                backspace_ratio=round(0.05 + (i % 5) * 0.02, 2),
                shortcut_rate=round(0.1 + (i % 3) * 0.05, 2),
                clicks_per_min=float(3 + i % 8),
                scroll_events_per_min=float(i % 6),
                mouse_distance=float(200 + i * 50),
                hour_of_day=hour,
                day_of_week=day_of_week,
                session_length_so_far=float(i * 5),
            )
        )

    return rows


def build_features_for_date(date: dt.date, data_dir: Path) -> Path:
    """Generate dummy features for *date*, validate, and write to parquet.

    Args:
        date: Calendar date to build features for.
        data_dir: Root of processed data (e.g. ``Path("data/processed")``).
            Output lands at ``data_dir/features_v1/date=YYYY-MM-DD/features.parquet``.

    Returns:
        Path of the written parquet file.

    Raises:
        ValueError: If generated data fails ``FeatureSchemaV1`` validation.
    """
    rows = generate_dummy_features(date)
    df = pd.DataFrame([r.model_dump() for r in rows])

    FeatureSchemaV1.validate_dataframe(df)

    out_path = (
        data_dir / f"features_v1/date={date.isoformat()}" / "features.parquet"
    )
    return write_parquet(df, out_path)
