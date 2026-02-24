"""Streamlit labeling UI for ground-truth collection.

Launch with::

    streamlit run src/taskclf/ui/labeling.py -- --data-dir data/processed

Provides four panels:

1. **Queue** -- pending ``LabelRequest`` items, sorted by confidence.
2. **Summary** -- aggregated feature stats for the selected block
   (privacy-safe: no raw titles).
3. **Label form** -- dropdown for ``CoreLabel``, confidence slider, submit.
4. **History** -- recently created labels with edit capability.
"""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

from taskclf.core.defaults import DEFAULT_DATA_DIR
from taskclf.core.store import read_parquet
from taskclf.core.types import CoreLabel, LabelSpan
from taskclf.labels.queue import ActiveLabelingQueue
from taskclf.labels.store import (
    append_label_span,
    generate_label_summary,
    read_label_spans,
)


def _parse_args() -> Path:
    data_dir = DEFAULT_DATA_DIR
    args = sys.argv[1:]
    for i, arg in enumerate(args):
        if arg == "--data-dir" and i + 1 < len(args):
            data_dir = args[i + 1]
    return Path(data_dir)


def _load_features(data_dir: Path, start: dt.datetime, end: dt.datetime) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    current = start.date()
    while current <= end.date():
        fp = data_dir / f"features_v1/date={current.isoformat()}" / "features.parquet"
        if fp.exists():
            frames.append(read_parquet(fp))
        current += dt.timedelta(days=1)
    if frames:
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame()


def main() -> None:
    data_dir = _parse_args()
    labels_path = data_dir / "labels_v1" / "labels.parquet"
    queue_path = data_dir / "labels_v1" / "queue.json"

    st.set_page_config(page_title="taskclf — Labeling", layout="wide")
    st.title("Task Label Collection")

    col_queue, col_main = st.columns([1, 2])

    # --- Queue panel ---
    with col_queue:
        st.header("Labeling Queue")

        queue = ActiveLabelingQueue(queue_path) if queue_path.exists() else None
        pending = queue.get_pending(limit=20) if queue is not None else []

        if not pending:
            st.info("No pending labeling requests.")
        else:
            options = {
                f"{r.bucket_start_ts:%Y-%m-%d %H:%M} — {r.reason} (conf={r.confidence:.2f})": r
                for r in pending
            }
            selected_key = st.selectbox(
                "Select a request", list(options.keys()), key="queue_select"
            )
            if selected_key and selected_key in options:
                sel = options[selected_key]
                st.markdown(
                    f"**Predicted:** {sel.predicted_label}  \n"
                    f"**Confidence:** {sel.confidence}  \n"
                    f"**Reason:** {sel.reason}"
                )
                if st.button("Skip this request", key="skip_btn"):
                    assert queue is not None
                    queue.mark_done(sel.request_id, status="skipped")
                    st.rerun()

    # --- Main panel ---
    with col_main:
        tab_label, tab_history = st.tabs(["New Label", "History"])

        # --- Labeling form ---
        with tab_label:
            st.header("Add Label Block")

            with st.form("label_form"):
                fc1, fc2 = st.columns(2)
                with fc1:
                    start_date = st.date_input("Start date", value=dt.date.today())
                    start_time = st.time_input("Start time", value=dt.time(9, 0))
                with fc2:
                    end_date = st.date_input("End date", value=dt.date.today())
                    end_time = st.time_input("End time", value=dt.time(10, 0))

                label = st.selectbox(
                    "Label",
                    [cl.value for cl in CoreLabel],
                    key="label_select",
                )
                user_id = st.text_input("User ID", value="default-user")
                confidence = st.slider(
                    "Confidence", min_value=0.0, max_value=1.0, value=0.8, step=0.05
                )

                submitted = st.form_submit_button("Save Label Block")

            if submitted and label is not None:
                start_ts = dt.datetime.combine(start_date, start_time)
                end_ts = dt.datetime.combine(end_date, end_time)

                if end_ts <= start_ts:
                    st.error("End time must be after start time.")
                else:
                    feat_df = _load_features(data_dir, start_ts, end_ts)
                    if not feat_df.empty:
                        summary = generate_label_summary(feat_df, start_ts, end_ts)
                        st.subheader("Block Summary")
                        sc1, sc2, sc3 = st.columns(3)
                        sc1.metric("Buckets", summary["total_buckets"])
                        sc2.metric("Sessions", summary["session_count"])
                        sc3.metric(
                            "Avg keys/min",
                            summary["mean_keys_per_min"]
                            if summary["mean_keys_per_min"] is not None
                            else "N/A",
                        )
                        if summary["top_apps"]:
                            st.markdown(
                                "**Top apps:** "
                                + ", ".join(
                                    f"{a['app_id']} ({a['buckets']})"
                                    for a in summary["top_apps"][:3]
                                )
                            )

                    span = LabelSpan(
                        start_ts=start_ts,
                        end_ts=end_ts,
                        label=label,
                        provenance="manual",
                        user_id=user_id,
                        confidence=confidence,
                    )
                    try:
                        append_label_span(span, labels_path)
                        st.success(
                            f"Saved: **{label}** "
                            f"[{start_ts:%H:%M} — {end_ts:%H:%M}]"
                        )
                        if queue is not None and pending:
                            sel_key = st.session_state.get("queue_select")
                            if sel_key and sel_key in options:
                                queue.mark_done(
                                    options[sel_key].request_id,
                                    status="labeled",
                                )
                    except ValueError as exc:
                        st.error(str(exc))

        # --- History panel ---
        with tab_history:
            st.header("Recent Labels")
            if labels_path.exists():
                spans = read_label_spans(labels_path)
                if spans:
                    rows = [
                        {
                            "Start": s.start_ts,
                            "End": s.end_ts,
                            "Label": s.label,
                            "User": s.user_id or "—",
                            "Confidence": s.confidence,
                            "Provenance": s.provenance,
                        }
                        for s in sorted(spans, key=lambda s: s.start_ts, reverse=True)[:50]
                    ]
                    st.dataframe(pd.DataFrame(rows), use_container_width=True)
                else:
                    st.info("No labels yet.")
            else:
                st.info("No labels file found.")


if __name__ == "__main__":
    main()
