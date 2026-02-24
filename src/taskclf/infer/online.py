"""Real-time prediction loop: poll ActivityWatch, predict, smooth, and report.

The online predictor never retrains -- it loads a frozen model bundle and
applies it to live data from a running ActivityWatch server instance.
"""

from __future__ import annotations

import csv
import logging
import time
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
from sklearn.preprocessing import LabelEncoder

from taskclf.core.defaults import (
    DEFAULT_AW_HOST,
    DEFAULT_BUCKET_SECONDS,
    DEFAULT_IDLE_GAP_SECONDS,
    DEFAULT_OUT_DIR,
    DEFAULT_POLL_SECONDS,
    DEFAULT_REJECT_THRESHOLD,
    DEFAULT_SMOOTH_WINDOW,
    DEFAULT_TITLE_SALT,
    MIXED_UNKNOWN,
)
from taskclf.core.model_io import ModelMetadata, load_model_bundle
from taskclf.core.types import LABEL_SET_V1, FeatureRow
from taskclf.infer.batch import write_segments_json
from taskclf.infer.smooth import Segment, rolling_majority, segmentize
from taskclf.train.lgbm import CATEGORICAL_COLUMNS, FEATURE_COLUMNS

logger = logging.getLogger(__name__)


class OnlinePredictor:
    """Stateful single-bucket predictor with rolling smoothing.

    Maintains a buffer of recent raw predictions and a running list of
    segments so that callers only need to feed one :class:`FeatureRow`
    at a time.
    """

    def __init__(
        self,
        model: lgb.Booster,
        metadata: ModelMetadata,
        *,
        cat_encoders: dict[str, LabelEncoder] | None = None,
        smooth_window: int = DEFAULT_SMOOTH_WINDOW,
        bucket_seconds: int = DEFAULT_BUCKET_SECONDS,
        reject_threshold: float | None = DEFAULT_REJECT_THRESHOLD,
    ) -> None:
        self._model = model
        self._metadata = metadata
        self._cat_encoders = cat_encoders or {}
        self._smooth_window = smooth_window
        self._bucket_seconds = bucket_seconds
        self._reject_threshold = reject_threshold

        self._le = LabelEncoder()
        self._le.fit(sorted(LABEL_SET_V1))

        self._raw_buffer: deque[str] = deque(maxlen=max(smooth_window, 1))
        self._bucket_ts_buffer: deque[datetime] = deque(maxlen=max(smooth_window, 1))
        self._all_bucket_ts: list[datetime] = []
        self._all_smoothed: list[str] = []

    def _encode_value(self, col: str, value: Any) -> float:
        """Encode a single feature value, handling categoricals and nulls."""
        if col in CATEGORICAL_COLUMNS:
            le = self._cat_encoders.get(col)
            if le is not None:
                str_val = str(value)
                if str_val in set(le.classes_):
                    return float(le.transform([str_val])[0])
            return -1.0
        return float(value) if value is not None else 0.0

    def predict_bucket(self, row: FeatureRow) -> str:
        """Predict a single bucket and return the smoothed label.

        The raw prediction is appended to an internal rolling buffer.
        Smoothing is applied over the buffer contents, and the smoothed
        label for the *latest* position is returned.  Internal segment
        state is updated accordingly.

        Args:
            row: A validated :class:`FeatureRow` for one time bucket.

        Returns:
            The smoothed predicted label string.
        """
        x = np.array(
            [[self._encode_value(c, getattr(row, c)) for c in FEATURE_COLUMNS]],
            dtype=np.float64,
        )
        proba = self._model.predict(x)
        confidence = float(proba.max(axis=1)[0])
        pred_idx = int(proba.argmax(axis=1)[0])
        raw_label: str = self._le.inverse_transform([pred_idx])[0]

        if self._reject_threshold is not None and confidence < self._reject_threshold:
            raw_label = MIXED_UNKNOWN

        self._raw_buffer.append(raw_label)
        self._bucket_ts_buffer.append(row.bucket_start_ts)

        smoothed = rolling_majority(list(self._raw_buffer), window=self._smooth_window)
        smoothed_label = smoothed[-1]

        self._all_bucket_ts.append(row.bucket_start_ts)
        self._all_smoothed.append(smoothed_label)

        return smoothed_label

    def get_segments(self) -> list[Segment]:
        """Return the running segment list built from all predictions so far."""
        if not self._all_bucket_ts:
            return []
        return segmentize(
            self._all_bucket_ts,
            self._all_smoothed,
            bucket_seconds=self._bucket_seconds,
        )


def _append_prediction_csv(path: Path, bucket_ts: datetime, label: str) -> None:
    """Append a single prediction row to a CSV file (creating it if needed)."""
    write_header = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["bucket_start_ts", "predicted_label"])
        writer.writerow([bucket_ts.isoformat(), label])


def run_online_loop(
    *,
    model_dir: Path,
    aw_host: str = DEFAULT_AW_HOST,
    poll_seconds: int = DEFAULT_POLL_SECONDS,
    smooth_window: int = DEFAULT_SMOOTH_WINDOW,
    title_salt: str = DEFAULT_TITLE_SALT,
    out_dir: Path = Path(DEFAULT_OUT_DIR),
    bucket_seconds: int = DEFAULT_BUCKET_SECONDS,
    idle_gap_seconds: float = DEFAULT_IDLE_GAP_SECONDS,
    reject_threshold: float | None = DEFAULT_REJECT_THRESHOLD,
) -> None:
    """Poll ActivityWatch, predict, smooth, and write results continuously.

    Runs until interrupted with ``KeyboardInterrupt`` (Ctrl+C).  On
    shutdown, writes final segments and prints a summary.

    Session state is tracked across poll cycles.  A new session starts
    when the gap since the last observed event exceeds
    *idle_gap_seconds*.

    Args:
        model_dir: Path to a trained model bundle directory.
        aw_host: Base URL of the ActivityWatch server.
        poll_seconds: Seconds between polling iterations.
        smooth_window: Rolling majority window size.
        title_salt: Salt for hashing window titles.
        out_dir: Directory for ``predictions.csv`` and ``segments.json``.
        bucket_seconds: Width of each time bucket in seconds.
        idle_gap_seconds: Minimum gap (seconds) that starts a new session.
        reject_threshold: If given, predictions with
            ``max(proba) < reject_threshold`` become ``Mixed/Unknown``.
    """
    from taskclf.adapters.activitywatch.client import (
        fetch_aw_events,
        fetch_aw_input_events,
        find_input_bucket_id,
        find_window_bucket_id,
    )
    from taskclf.features.build import build_features_from_aw_events

    model, metadata, cat_encoders = load_model_bundle(Path(model_dir))
    logger.info("Loaded model from %s (schema=%s)", model_dir, metadata.schema_hash)

    bucket_id = find_window_bucket_id(aw_host)
    logger.info("Using AW window bucket: %s", bucket_id)

    input_bucket_id = find_input_bucket_id(aw_host)
    if input_bucket_id:
        logger.info("Using AW input bucket: %s", input_bucket_id)
    else:
        logger.info("No aw-watcher-input bucket found; keyboard/mouse features will be None")

    predictor = OnlinePredictor(
        model,
        metadata,
        cat_encoders=cat_encoders,
        smooth_window=smooth_window,
        bucket_seconds=bucket_seconds,
        reject_threshold=reject_threshold,
    )

    pred_path = out_dir / "predictions.csv"
    seg_path = out_dir / "segments.json"

    session_start: datetime | None = None
    last_event_ts: datetime | None = None
    idle_gap = timedelta(seconds=idle_gap_seconds)

    print(f"Online inference started (polling every {poll_seconds}s, bucket={bucket_id})")
    if input_bucket_id:
        print(f"Input watcher active: {input_bucket_id}")
    print("Press Ctrl+C to stop.\n")

    try:
        while True:
            now = datetime.now(timezone.utc).replace(tzinfo=None)
            window_start = now - timedelta(seconds=poll_seconds)

            try:
                events = fetch_aw_events(
                    aw_host, bucket_id, window_start, now, title_salt=title_salt,
                )
            except Exception:
                logger.warning("Failed to fetch AW events, will retry", exc_info=True)
                time.sleep(poll_seconds)
                continue

            if not events:
                logger.debug("No events in window [%s, %s)", window_start, now)
                time.sleep(poll_seconds)
                continue

            input_events = None
            if input_bucket_id:
                try:
                    input_events = fetch_aw_input_events(
                        aw_host, input_bucket_id, window_start, now,
                    ) or None
                except Exception:
                    logger.warning("Failed to fetch input events", exc_info=True)

            earliest_new = min(ev.timestamp for ev in events)
            if last_event_ts is not None and earliest_new - last_event_ts >= idle_gap:
                session_start = earliest_new
                logger.info("New session started at %s", session_start)

            if session_start is None:
                session_start = earliest_new

            rows = build_features_from_aw_events(
                events,
                input_events=input_events,
                bucket_seconds=bucket_seconds,
                session_start=session_start,
            )
            if not rows:
                time.sleep(poll_seconds)
                continue

            last_event_ts = max(ev.timestamp for ev in events)

            for row in rows:
                label = predictor.predict_bucket(row)
                ts_str = row.bucket_start_ts.strftime("%H:%M")
                print(f"[{ts_str}] {label}")
                _append_prediction_csv(pred_path, row.bucket_start_ts, label)

            segments = predictor.get_segments()
            write_segments_json(segments, seg_path)

            time.sleep(poll_seconds)

    except KeyboardInterrupt:
        print("\nShutting down...")

    segments = predictor.get_segments()
    if segments:
        write_segments_json(segments, seg_path)
        print(f"Final segments written to {seg_path} ({len(segments)} segments)")

        try:
            from taskclf.report.daily import build_daily_report
            from taskclf.report.export import export_report_json

            report = build_daily_report(segments, bucket_seconds=bucket_seconds)
            report_path = export_report_json(report, out_dir / f"report_{report.date}.json")
            print(f"Daily report written to {report_path}")
        except Exception:
            logger.warning("Could not generate daily report", exc_info=True)
    else:
        print("No predictions were made.")
