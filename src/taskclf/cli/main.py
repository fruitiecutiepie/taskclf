"""Typer CLI entrypoint and command definitions for taskclf."""

import datetime as dt
from pathlib import Path

import typer

from taskclf.core.defaults import (
    DEFAULT_AW_HOST,
    DEFAULT_DATA_DIR,
    DEFAULT_MODELS_DIR,
    DEFAULT_NUM_BOOST_ROUND,
    DEFAULT_OUT_DIR,
    DEFAULT_POLL_SECONDS,
    DEFAULT_RAW_AW_DIR,
    DEFAULT_SMOOTH_WINDOW,
    DEFAULT_TITLE_SALT,
)

app = typer.Typer()

# -- ingest -------------------------------------------------------------------
ingest_app = typer.Typer()
app.add_typer(ingest_app, name="ingest")


@ingest_app.command("aw")
def ingest_aw_cmd(
    input_file: str = typer.Option(..., "--input", help="Path to an ActivityWatch export JSON file"),
    out_dir: str = typer.Option(DEFAULT_RAW_AW_DIR, help="Output directory for normalized events (partitioned by date)"),
    title_salt: str = typer.Option(DEFAULT_TITLE_SALT, "--title-salt", help="Salt for hashing window titles"),
) -> None:
    """Ingest an ActivityWatch JSON export into privacy-safe normalized events."""
    from collections import defaultdict

    import pandas as pd

    from taskclf.adapters.activitywatch.client import (
        parse_aw_export,
        parse_aw_input_export,
    )
    from taskclf.core.store import write_parquet

    input_path = Path(input_file)
    if not input_path.exists():
        typer.echo(f"File not found: {input_path}", err=True)
        raise typer.Exit(code=1)

    events = parse_aw_export(input_path, title_salt=title_salt)
    if not events:
        typer.echo("No window events found in the export file.", err=True)
        raise typer.Exit(code=1)

    by_date: dict[str, list] = defaultdict(list)
    for ev in events:
        day = ev.timestamp.date().isoformat()
        by_date[day].append(ev.model_dump())

    out_base = Path(out_dir)
    for day, rows in sorted(by_date.items()):
        df = pd.DataFrame(rows)
        out_path = out_base / day / "events.parquet"
        write_parquet(df, out_path)
        typer.echo(f"  {day}: {len(rows)} events -> {out_path}")

    unique_apps = {ev.app_id for ev in events}
    date_range = f"{events[0].timestamp.date()} to {events[-1].timestamp.date()}"
    typer.echo(
        f"Ingested {len(events)} events across {len(by_date)} day(s) "
        f"({date_range}), {len(unique_apps)} unique apps"
    )

    # Ingest input events (aw-watcher-input) if present
    input_events = parse_aw_input_export(input_path)
    if input_events:
        input_by_date: dict[str, list] = defaultdict(list)
        for ie in input_events:
            day = ie.timestamp.date().isoformat()
            input_by_date[day].append(ie.model_dump())

        input_out_base = Path(out_dir).parent / "aw-input"
        for day, rows in sorted(input_by_date.items()):
            df = pd.DataFrame(rows)
            out_path = input_out_base / day / "events.parquet"
            write_parquet(df, out_path)
            typer.echo(f"  {day}: {len(rows)} input events -> {out_path}")

        ie_range = (
            f"{input_events[0].timestamp.date()} to "
            f"{input_events[-1].timestamp.date()}"
        )
        typer.echo(
            f"Ingested {len(input_events)} input events across "
            f"{len(input_by_date)} day(s) ({ie_range})"
        )


# -- features ----------------------------------------------------------------
features_app = typer.Typer()
app.add_typer(features_app, name="features")


@features_app.command("build")
def features_build(
    date: str = typer.Option(..., help="Date in YYYY-MM-DD format"),
    data_dir: str = typer.Option(DEFAULT_DATA_DIR, help="Processed data directory"),
) -> None:
    """Build per-minute feature rows for a given date and write to parquet."""
    from taskclf.features.build import build_features_for_date

    parsed_date = dt.date.fromisoformat(date)
    out_path = build_features_for_date(parsed_date, Path(data_dir))
    typer.echo(f"Wrote features to {out_path}")


# -- labels -------------------------------------------------------------------
labels_app = typer.Typer()
app.add_typer(labels_app, name="labels")


@labels_app.command("import")
def labels_import_cmd(
    file: str = typer.Option(..., "--file", help="Path to labels CSV (start_ts, end_ts, label, provenance)"),
    data_dir: str = typer.Option(DEFAULT_DATA_DIR, help="Processed data directory"),
) -> None:
    """Import label spans from a CSV file and write to parquet."""
    from taskclf.labels.store import import_labels_from_csv, write_label_spans

    csv_path = Path(file)
    if not csv_path.exists():
        typer.echo(f"File not found: {csv_path}", err=True)
        raise typer.Exit(code=1)

    spans = import_labels_from_csv(csv_path)
    typer.echo(f"Validated {len(spans)} label spans")

    out_path = Path(data_dir) / "labels_v1" / "labels.parquet"
    write_label_spans(spans, out_path)
    typer.echo(f"Wrote labels to {out_path}")


# -- train --------------------------------------------------------------------
train_app = typer.Typer()
app.add_typer(train_app, name="train")


@train_app.command("lgbm")
def train_lgbm_cmd(
    date_from: str = typer.Option(..., "--from", help="Start date (YYYY-MM-DD)"),
    date_to: str = typer.Option(..., "--to", help="End date (YYYY-MM-DD, inclusive)"),
    synthetic: bool = typer.Option(False, "--synthetic", help="Generate dummy features + labels instead of reading from disk"),
    models_dir: str = typer.Option(DEFAULT_MODELS_DIR, help="Base directory for model bundles"),
    data_dir: str = typer.Option(DEFAULT_DATA_DIR, help="Processed data directory"),
    num_boost_round: int = typer.Option(DEFAULT_NUM_BOOST_ROUND, help="Number of boosting rounds"),
) -> None:
    """Train a LightGBM multiclass model and save the model bundle."""
    import pandas as pd

    from taskclf.core.model_io import build_metadata, save_model_bundle
    from taskclf.core.store import read_parquet
    from taskclf.features.build import generate_dummy_features
    from taskclf.labels.store import generate_dummy_labels
    from taskclf.train.dataset import assign_labels_to_buckets, split_by_day
    from taskclf.train.lgbm import train_lgbm

    start = dt.date.fromisoformat(date_from)
    end = dt.date.fromisoformat(date_to)

    all_features: list[pd.DataFrame] = []
    all_labels = []
    current = start

    while current <= end:
        if synthetic:
            rows = generate_dummy_features(current, n_rows=60)
            df = pd.DataFrame([r.model_dump() for r in rows])
            labels = generate_dummy_labels(current, n_rows=60)
        else:
            parquet_path = Path(data_dir) / f"features_v1/date={current.isoformat()}" / "features.parquet"
            if not parquet_path.exists():
                typer.echo(f"  skipping {current} (no features file)")
                current += dt.timedelta(days=1)
                continue
            df = read_parquet(parquet_path)
            labels = generate_dummy_labels(current, n_rows=len(df))

        all_features.append(df)
        all_labels.extend(labels)
        current += dt.timedelta(days=1)

    if not all_features:
        typer.echo("No feature data found for the given date range.", err=True)
        raise typer.Exit(code=1)

    features_df = pd.concat(all_features, ignore_index=True)
    typer.echo(f"Loaded {len(features_df)} feature rows across {len(all_features)} day(s)")

    labeled_df = assign_labels_to_buckets(features_df, all_labels)
    typer.echo(f"Labeled {len(labeled_df)} / {len(features_df)} rows")

    if labeled_df.empty:
        typer.echo("No labeled rows — cannot train.", err=True)
        raise typer.Exit(code=1)

    train_df, val_df = split_by_day(labeled_df)
    typer.echo(f"Train: {len(train_df)} rows, Val: {len(val_df)} rows")

    model, metrics, cm_df, params, cat_encoders = train_lgbm(
        train_df, val_df, num_boost_round=num_boost_round,
    )
    typer.echo(f"Macro F1: {metrics['macro_f1']}")

    metadata = build_metadata(
        label_set=metrics["label_names"],
        train_date_from=start,
        train_date_to=end,
        params=params,
    )

    run_dir = save_model_bundle(
        model=model,
        metadata=metadata,
        metrics=metrics,
        confusion_df=cm_df,
        base_dir=Path(models_dir),
        cat_encoders=cat_encoders,
    )
    typer.echo(f"Model bundle saved to {run_dir}")


# -- infer --------------------------------------------------------------------
infer_app = typer.Typer()
app.add_typer(infer_app, name="infer")


@infer_app.command("batch")
def infer_batch_cmd(
    model_dir: str = typer.Option(..., "--model-dir", help="Path to a model run directory"),
    date_from: str = typer.Option(..., "--from", help="Start date (YYYY-MM-DD)"),
    date_to: str = typer.Option(..., "--to", help="End date (YYYY-MM-DD, inclusive)"),
    synthetic: bool = typer.Option(False, "--synthetic", help="Generate dummy features instead of reading from disk"),
    data_dir: str = typer.Option(DEFAULT_DATA_DIR, help="Processed data directory"),
    out_dir: str = typer.Option(DEFAULT_OUT_DIR, help="Output directory for predictions and segments"),
    smooth_window: int = typer.Option(DEFAULT_SMOOTH_WINDOW, help="Rolling majority smoothing window size"),
) -> None:
    """Run batch inference: predict, smooth, and segmentize."""
    import pandas as pd

    from taskclf.core.model_io import load_model_bundle
    from taskclf.core.store import read_parquet
    from taskclf.features.build import generate_dummy_features
    from taskclf.infer.batch import (
        run_batch_inference,
        write_predictions_csv,
        write_segments_json,
    )

    model, metadata, cat_encoders = load_model_bundle(Path(model_dir))
    typer.echo(f"Loaded model from {model_dir} (schema={metadata.schema_hash})")

    start = dt.date.fromisoformat(date_from)
    end = dt.date.fromisoformat(date_to)

    all_features: list[pd.DataFrame] = []
    current = start
    while current <= end:
        if synthetic:
            rows = generate_dummy_features(current, n_rows=60)
            df = pd.DataFrame([r.model_dump() for r in rows])
        else:
            parquet_path = Path(data_dir) / f"features_v1/date={current.isoformat()}" / "features.parquet"
            if not parquet_path.exists():
                typer.echo(f"  skipping {current} (no features file)")
                current += dt.timedelta(days=1)
                continue
            df = read_parquet(parquet_path)
        all_features.append(df)
        current += dt.timedelta(days=1)

    if not all_features:
        typer.echo("No feature data found for the given date range.", err=True)
        raise typer.Exit(code=1)

    features_df = pd.concat(all_features, ignore_index=True)
    features_df = features_df.sort_values("bucket_start_ts").reset_index(drop=True)
    typer.echo(f"Loaded {len(features_df)} feature rows")

    smoothed_labels, segments = run_batch_inference(
        model, features_df, cat_encoders=cat_encoders, smooth_window=smooth_window,
    )
    typer.echo(f"Predicted {len(smoothed_labels)} buckets -> {len(segments)} segments")

    out = Path(out_dir)
    pred_path = write_predictions_csv(features_df, smoothed_labels, out / "predictions.csv")
    seg_path = write_segments_json(segments, out / "segments.json")

    typer.echo(f"Predictions: {pred_path}")
    typer.echo(f"Segments:    {seg_path}")


@infer_app.command("online")
def infer_online_cmd(
    model_dir: str = typer.Option(..., "--model-dir", help="Path to a model run directory"),
    poll_seconds: int = typer.Option(DEFAULT_POLL_SECONDS, "--poll-seconds", help="Seconds between polling iterations"),
    aw_host: str = typer.Option(DEFAULT_AW_HOST, "--aw-host", help="ActivityWatch server URL"),
    smooth_window: int = typer.Option(DEFAULT_SMOOTH_WINDOW, "--smooth-window", help="Rolling majority smoothing window size"),
    title_salt: str = typer.Option(DEFAULT_TITLE_SALT, "--title-salt", help="Salt for hashing window titles"),
    out_dir: str = typer.Option(DEFAULT_OUT_DIR, help="Output directory for predictions and segments"),
) -> None:
    """Run online inference: poll ActivityWatch, predict, smooth, and report."""
    from taskclf.infer.online import run_online_loop

    run_online_loop(
        model_dir=Path(model_dir),
        aw_host=aw_host,
        poll_seconds=poll_seconds,
        smooth_window=smooth_window,
        title_salt=title_salt,
        out_dir=Path(out_dir),
    )


# -- report -------------------------------------------------------------------
report_app = typer.Typer()
app.add_typer(report_app, name="report")


@report_app.command("daily")
def report_daily_cmd(
    segments_file: str = typer.Option(..., "--segments-file", help="Path to segments.json"),
    out_dir: str = typer.Option(DEFAULT_OUT_DIR, help="Output directory for report files"),
) -> None:
    """Generate a daily report from a segments JSON file."""
    from taskclf.infer.batch import read_segments_json
    from taskclf.report.daily import build_daily_report
    from taskclf.report.export import export_report_json

    seg_path = Path(segments_file)
    if not seg_path.exists():
        typer.echo(f"Segments file not found: {seg_path}", err=True)
        raise typer.Exit(code=1)

    segments = read_segments_json(seg_path)
    typer.echo(f"Loaded {len(segments)} segments")

    report = build_daily_report(segments)
    typer.echo(f"Report for {report.date}: {report.total_minutes} minutes across {report.segments_count} segments")

    out = Path(out_dir)
    report_path = export_report_json(report, out / f"report_{report.date}.json")
    typer.echo(f"Report written to {report_path}")


if __name__ == "__main__":
    app()
