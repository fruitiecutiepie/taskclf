"""Real-time prediction loop: poll ActivityWatch, predict, smooth, and report.

The online predictor never retrains -- it loads a frozen model bundle and
applies it to live data from a running ActivityWatch server instance.
"""

from __future__ import annotations

import csv
import json
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
    DEFAULT_LABEL_CONFIDENCE_THRESHOLD,
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
from taskclf.infer.calibration import Calibrator, CalibratorStore, IdentityCalibrator
from taskclf.infer.prediction import WindowPrediction
from taskclf.infer.smooth import Segment, merge_short_segments, rolling_majority, segmentize
from taskclf.infer.taxonomy import TaxonomyConfig, TaxonomyResolver
from taskclf.train.lgbm import CATEGORICAL_COLUMNS, FEATURE_COLUMNS

logger = logging.getLogger(__name__)

_SORTED_LABELS = sorted(LABEL_SET_V1)


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
        taxonomy: TaxonomyConfig | None = None,
        calibrator: Calibrator | None = None,
        calibrator_store: CalibratorStore | None = None,
    ) -> None:
        self._model = model
        self._metadata = metadata
        self._cat_encoders = cat_encoders or {}
        self._smooth_window = smooth_window
        self._bucket_seconds = bucket_seconds
        self._reject_threshold = reject_threshold

        self._le = LabelEncoder()
        self._le.fit(_SORTED_LABELS)

        self._resolver: TaxonomyResolver | None = None
        if taxonomy is not None:
            self._resolver = TaxonomyResolver(taxonomy)

        self._calibrator: Calibrator = calibrator or IdentityCalibrator()
        self._calibrator_store: CalibratorStore | None = calibrator_store

        self._raw_buffer: deque[str] = deque(maxlen=max(smooth_window, 1))
        self._bucket_ts_buffer: deque[datetime] = deque(maxlen=max(smooth_window, 1))
        self._all_bucket_ts: list[datetime] = []
        self._all_raw: list[str] = []
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

    def predict_bucket(self, row: FeatureRow) -> WindowPrediction:
        """Predict a single bucket and return a full :class:`WindowPrediction`.

        Pipeline: raw model proba -> calibrate -> reject check ->
        rolling majority smoothing (on core labels) -> taxonomy resolve
        -> assemble ``WindowPrediction``.

        Args:
            row: A validated :class:`FeatureRow` for one time bucket.

        Returns:
            A :class:`WindowPrediction` containing core and mapped
            predictions, confidence, and rejection status.
        """
        x = np.array(
            [[self._encode_value(c, getattr(row, c)) for c in FEATURE_COLUMNS]],
            dtype=np.float64,
        )
        raw_proba = self._model.predict(x)

        if self._calibrator_store is not None:
            cal = self._calibrator_store.get_calibrator(row.user_id)
        else:
            cal = self._calibrator
        calibrated = cal.calibrate(raw_proba)
        proba_vec: np.ndarray = calibrated[0]

        confidence = float(proba_vec.max())
        pred_idx = int(proba_vec.argmax())
        core_label_name: str = self._le.inverse_transform([pred_idx])[0]

        is_rejected = (
            self._reject_threshold is not None
            and confidence < self._reject_threshold
        )

        smoothing_label = MIXED_UNKNOWN if is_rejected else core_label_name
        self._raw_buffer.append(smoothing_label)
        self._bucket_ts_buffer.append(row.bucket_start_ts)

        smoothed = rolling_majority(list(self._raw_buffer), window=self._smooth_window)
        smoothed_label = smoothed[-1]

        self._all_bucket_ts.append(row.bucket_start_ts)
        self._all_raw.append(smoothing_label)
        self._all_smoothed.append(smoothed_label)

        if self._resolver is not None:
            tax_result = self._resolver.resolve(
                pred_idx, proba_vec, is_rejected=is_rejected,
            )
            mapped_label_name = tax_result.mapped_label
            mapped_probs = tax_result.mapped_probs
        else:
            mapped_label_name = smoothed_label
            mapped_probs = {
                lbl: float(proba_vec[i]) for i, lbl in enumerate(_SORTED_LABELS)
            }

        return WindowPrediction(
            user_id=row.user_id,
            bucket_start_ts=row.bucket_start_ts,
            core_label_id=pred_idx,
            core_label_name=core_label_name,
            core_probs=[round(float(p), 6) for p in proba_vec],
            confidence=round(confidence, 6),
            is_rejected=is_rejected,
            mapped_label_name=mapped_label_name,
            mapped_probs=mapped_probs,
            model_version=self._metadata.schema_hash,
            schema_version="features_v1",
            label_version="labels_v1",
        )

    def get_segments(self) -> list[Segment]:
        """Return the running segment list built from all predictions so far.

        Applies hysteresis merging so segments shorter than
        ``MIN_BLOCK_DURATION_SECONDS`` are absorbed by their neighbours.
        """
        if not self._all_bucket_ts:
            return []
        segments = segmentize(
            self._all_bucket_ts,
            self._all_smoothed,
            bucket_seconds=self._bucket_seconds,
        )
        return merge_short_segments(segments, bucket_seconds=self._bucket_seconds)


_PREDICTION_CSV_COLUMNS = [
    "bucket_start_ts",
    "core_label",
    "confidence",
    "is_rejected",
    "mapped_label",
    "core_probs",
    "mapped_probs",
    "model_version",
]


def _append_prediction_csv(path: Path, prediction: WindowPrediction) -> None:
    """Append a single :class:`WindowPrediction` row to a CSV file."""
    write_header = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(_PREDICTION_CSV_COLUMNS)
        writer.writerow([
            prediction.bucket_start_ts.isoformat(),
            prediction.core_label_name,
            f"{prediction.confidence:.4f}",
            prediction.is_rejected,
            prediction.mapped_label_name,
            json.dumps([round(p, 4) for p in prediction.core_probs]),
            json.dumps({k: round(v, 4) for k, v in prediction.mapped_probs.items()}),
            prediction.model_version,
        ])


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
    taxonomy_path: Path | None = None,
    calibrator_path: Path | None = None,
    calibrator_store_path: Path | None = None,
    label_queue_path: Path | None = None,
    label_confidence_threshold: float = DEFAULT_LABEL_CONFIDENCE_THRESHOLD,
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
        taxonomy_path: Optional path to a taxonomy YAML file.  When
            provided, output labels are mapped to user-defined buckets.
        calibrator_path: Optional path to a calibrator JSON file.  When
            provided, raw model probabilities are calibrated before the
            reject decision.
        calibrator_store_path: Optional path to a calibrator store
            directory.  When provided, per-user calibration is applied.
            Takes precedence over *calibrator_path*.
        label_queue_path: Optional path to the labeling queue JSON.
            When provided, low-confidence predictions are auto-enqueued
            for manual labeling.
        label_confidence_threshold: Predictions with confidence below
            this value are enqueued when *label_queue_path* is set.
    """
    from taskclf.adapters.activitywatch.client import (
        fetch_aw_events,
        fetch_aw_input_events,
        find_input_bucket_id,
        find_window_bucket_id,
    )
    from taskclf.features.build import build_features_from_aw_events
    from taskclf.infer.calibration import load_calibrator, load_calibrator_store
    from taskclf.infer.taxonomy import load_taxonomy

    taxonomy: TaxonomyConfig | None = None
    if taxonomy_path is not None:
        taxonomy = load_taxonomy(taxonomy_path)
        logger.info("Loaded taxonomy from %s (user=%s)", taxonomy_path, taxonomy.user_id)

    calibrator: Calibrator | None = None
    if calibrator_path is not None:
        calibrator = load_calibrator(calibrator_path)
        logger.info("Loaded calibrator from %s", calibrator_path)

    cal_store: CalibratorStore | None = None
    if calibrator_store_path is not None:
        cal_store = load_calibrator_store(calibrator_store_path)
        logger.info(
            "Loaded calibrator store from %s (%d per-user calibrators)",
            calibrator_store_path, len(cal_store.user_calibrators),
        )

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
        taxonomy=taxonomy,
        calibrator=calibrator,
        calibrator_store=cal_store,
    )

    label_queue = None
    if label_queue_path is not None:
        from taskclf.labels.queue import ActiveLabelingQueue

        label_queue = ActiveLabelingQueue(label_queue_path)
        logger.info(
            "Label queue active: %s (threshold=%.2f)",
            label_queue_path, label_confidence_threshold,
        )

    pred_path = out_dir / "predictions.csv"
    seg_path = out_dir / "segments.json"

    session_start: datetime | None = None
    last_event_ts: datetime | None = None
    idle_gap = timedelta(seconds=idle_gap_seconds)
    total_enqueued = 0

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
                prediction = predictor.predict_bucket(row)
                ts_str = row.bucket_start_ts.strftime("%H:%M")
                conf_str = f"{prediction.confidence:.2f}"
                print(f"[{ts_str}] {prediction.mapped_label_name} (confidence: {conf_str})")
                _append_prediction_csv(pred_path, prediction)

                if (
                    label_queue is not None
                    and prediction.confidence < label_confidence_threshold
                ):
                    import pandas as pd

                    enqueue_df = pd.DataFrame([{
                        "user_id": prediction.user_id or "default-user",
                        "bucket_start_ts": prediction.bucket_start_ts,
                        "bucket_end_ts": prediction.bucket_start_ts + timedelta(seconds=bucket_seconds),
                        "confidence": prediction.confidence,
                        "predicted_label": prediction.core_label_name,
                    }])
                    n = label_queue.enqueue_low_confidence(
                        enqueue_df, threshold=label_confidence_threshold,
                    )
                    if n > 0:
                        total_enqueued += n
                        print(f"  → enqueued for labeling (conf={conf_str})")

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

            report = build_daily_report(
                segments,
                bucket_seconds=bucket_seconds,
                raw_labels=predictor._all_raw or None,
                smoothed_labels=predictor._all_smoothed or None,
            )
            report_path = export_report_json(report, out_dir / f"report_{report.date}.json")
            print(f"Daily report written to {report_path}")
            if report.flap_rate_raw is not None:
                print(
                    f"Flap rate: raw={report.flap_rate_raw:.4f}  "
                    f"smoothed={report.flap_rate_smoothed:.4f}"
                )
        except Exception:
            logger.warning("Could not generate daily report", exc_info=True)

        try:
            from taskclf.core.telemetry import TelemetryStore, compute_telemetry

            import pandas as pd

            ts_records = [
                {"bucket_start_ts": ts} for ts in predictor._all_bucket_ts
            ]
            ts_df = pd.DataFrame(ts_records) if ts_records else pd.DataFrame()

            if not ts_df.empty:
                snapshot = compute_telemetry(
                    ts_df,
                    labels=predictor._all_smoothed or None,
                )
                store = TelemetryStore(out_dir / "telemetry")
                store_path = store.append(snapshot)
                print(f"Telemetry snapshot written to {store_path}")
        except Exception:
            logger.warning("Could not write telemetry snapshot", exc_info=True)
    else:
        print("No predictions were made.")

    if total_enqueued > 0:
        print(f"Enqueued {total_enqueued} low-confidence bucket(s) for labeling.")
