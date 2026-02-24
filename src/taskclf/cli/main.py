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
    DEFAULT_REJECT_THRESHOLD,
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


@labels_app.command("add-block")
def labels_add_block_cmd(
    start: str = typer.Option(..., "--start", help="Block start timestamp (ISO-8601 UTC)"),
    end: str = typer.Option(..., "--end", help="Block end timestamp (ISO-8601 UTC)"),
    label: str = typer.Option(..., "--label", help="Core label (Build, Debug, Review, Write, ReadResearch, Communicate, Meet, BreakIdle)"),
    user_id: str = typer.Option("default-user", "--user-id", help="User ID for this label"),
    confidence: float | None = typer.Option(None, "--confidence", min=0.0, max=1.0, help="Labeler confidence (0-1)"),
    data_dir: str = typer.Option(DEFAULT_DATA_DIR, help="Processed data directory"),
    model_dir: str | None = typer.Option(None, "--model-dir", help="Model bundle directory (for predicted label display)"),
) -> None:
    """Create a manual label block for a time range."""
    import pandas as pd
    from rich.console import Console
    from rich.table import Table

    from taskclf.core.store import read_parquet
    from taskclf.core.types import LabelSpan
    from taskclf.labels.store import append_label_span, generate_label_summary

    console = Console()

    start_ts = dt.datetime.fromisoformat(start)
    end_ts = dt.datetime.fromisoformat(end)

    span = LabelSpan(
        start_ts=start_ts,
        end_ts=end_ts,
        label=label,
        provenance="manual",
        user_id=user_id,
        confidence=confidence,
    )

    features_dfs: list[pd.DataFrame] = []
    data_path = Path(data_dir)
    current_date = start_ts.date()
    while current_date <= end_ts.date():
        fp = data_path / f"features_v1/date={current_date.isoformat()}" / "features.parquet"
        if fp.exists():
            features_dfs.append(read_parquet(fp))
        current_date += dt.timedelta(days=1)

    if features_dfs:
        feat_df = pd.concat(features_dfs, ignore_index=True)
        summary = generate_label_summary(feat_df, start_ts, end_ts)

        table = Table(title="Block Summary")
        table.add_column("Metric", style="bold")
        table.add_column("Value")
        table.add_row("Time range", f"{start_ts} → {end_ts}")
        table.add_row("Buckets", str(summary["total_buckets"]))
        table.add_row("Sessions", str(summary["session_count"]))
        if summary["mean_keys_per_min"] is not None:
            table.add_row("Avg keys/min", str(summary["mean_keys_per_min"]))
        if summary["mean_clicks_per_min"] is not None:
            table.add_row("Avg clicks/min", str(summary["mean_clicks_per_min"]))
        for app_info in summary["top_apps"][:3]:
            table.add_row(f"App: {app_info['app_id']}", f"{app_info['buckets']} buckets")
        console.print(table)

    if model_dir is not None and features_dfs:
        try:
            from taskclf.core.model_io import load_model_bundle
            from taskclf.infer.batch import predict_labels

            model, _meta, cat_encoders = load_model_bundle(Path(model_dir))
            from sklearn.preprocessing import LabelEncoder

            from taskclf.core.types import LABEL_SET_V1

            le = LabelEncoder()
            le.fit(sorted(LABEL_SET_V1))
            preds = predict_labels(model, feat_df, le, cat_encoders=cat_encoders)
            from collections import Counter

            top_pred = Counter(preds).most_common(1)[0][0]
            console.print(f"[bold]Top predicted label:[/bold] {top_pred}")
        except Exception as exc:
            console.print(f"[dim]Could not predict: {exc}[/dim]")

    labels_path = Path(data_dir) / "labels_v1" / "labels.parquet"
    try:
        append_label_span(span, labels_path)
    except ValueError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1)

    typer.echo(
        f"Added label block: {label} [{start_ts} → {end_ts}] "
        f"user={user_id} confidence={confidence}"
    )


@labels_app.command("show-queue")
def labels_show_queue_cmd(
    user_id: str | None = typer.Option(None, "--user-id", help="Filter to a specific user"),
    limit: int = typer.Option(10, "--limit", help="Maximum items to show"),
    data_dir: str = typer.Option(DEFAULT_DATA_DIR, help="Processed data directory"),
) -> None:
    """Show pending items in the active labeling queue."""
    from rich.console import Console
    from rich.table import Table

    from taskclf.labels.queue import ActiveLabelingQueue

    console = Console()
    queue_path = Path(data_dir) / "labels_v1" / "queue.json"

    if not queue_path.exists():
        console.print("[dim]No labeling queue found.[/dim]")
        return

    queue = ActiveLabelingQueue(queue_path)
    pending = queue.get_pending(user_id=user_id, limit=limit)

    if not pending:
        console.print("[dim]No pending labeling requests.[/dim]")
        return

    table = Table(title=f"Pending Labeling Requests ({len(pending)})")
    table.add_column("ID", style="dim", max_width=8)
    table.add_column("User")
    table.add_column("Time Range")
    table.add_column("Reason")
    table.add_column("Predicted")
    table.add_column("Confidence", justify="right")

    for req in pending:
        table.add_row(
            req.request_id[:8],
            req.user_id,
            f"{req.bucket_start_ts:%H:%M} → {req.bucket_end_ts:%H:%M}",
            req.reason,
            req.predicted_label or "—",
            f"{req.confidence:.2f}" if req.confidence is not None else "—",
        )

    console.print(table)


@labels_app.command("project")
def labels_project_cmd(
    date_from: str = typer.Option(..., "--from", help="Start date (YYYY-MM-DD)"),
    date_to: str = typer.Option(..., "--to", help="End date (YYYY-MM-DD, inclusive)"),
    data_dir: str = typer.Option(DEFAULT_DATA_DIR, help="Processed data directory"),
    out_dir: str = typer.Option(DEFAULT_DATA_DIR, "--out-dir", help="Output directory for projected labels"),
) -> None:
    """Project label blocks onto feature windows using strict containment rules."""
    import pandas as pd

    from taskclf.core.store import read_parquet, write_parquet
    from taskclf.labels.projection import project_blocks_to_windows
    from taskclf.labels.store import read_label_spans

    labels_path = Path(data_dir) / "labels_v1" / "labels.parquet"
    if not labels_path.exists():
        typer.echo(f"Labels file not found: {labels_path}", err=True)
        raise typer.Exit(code=1)

    spans = read_label_spans(labels_path)
    typer.echo(f"Loaded {len(spans)} label spans")

    start = dt.date.fromisoformat(date_from)
    end = dt.date.fromisoformat(date_to)

    all_features: list[pd.DataFrame] = []
    current = start
    while current <= end:
        fp = Path(data_dir) / f"features_v1/date={current.isoformat()}" / "features.parquet"
        if fp.exists():
            all_features.append(read_parquet(fp))
        else:
            typer.echo(f"  skipping {current} (no features file)")
        current += dt.timedelta(days=1)

    if not all_features:
        typer.echo("No feature data found for the given date range.", err=True)
        raise typer.Exit(code=1)

    features_df = pd.concat(all_features, ignore_index=True)
    projected = project_blocks_to_windows(features_df, spans)

    out_path = Path(out_dir) / "labels_v1" / "projected_labels.parquet"
    write_parquet(projected, out_path)
    typer.echo(
        f"Projected {len(projected)} labeled windows "
        f"(from {len(features_df)} total) -> {out_path}"
    )


# -- train --------------------------------------------------------------------
train_app = typer.Typer()
app.add_typer(train_app, name="train")


@train_app.command("build-dataset")
def train_build_dataset_cmd(
    date_from: str = typer.Option(..., "--from", help="Start date (YYYY-MM-DD)"),
    date_to: str = typer.Option(..., "--to", help="End date (YYYY-MM-DD, inclusive)"),
    synthetic: bool = typer.Option(False, "--synthetic", help="Generate dummy features + labels"),
    data_dir: str = typer.Option(DEFAULT_DATA_DIR, help="Processed data directory"),
    out_dir: str = typer.Option(DEFAULT_DATA_DIR, "--out-dir", help="Output directory for X/y/splits"),
    holdout_fraction: float = typer.Option(0.0, "--holdout-fraction", help="Fraction of users to hold out for test"),
    train_ratio: float = typer.Option(0.70, "--train-ratio", help="Chronological train fraction per user"),
    val_ratio: float = typer.Option(0.15, "--val-ratio", help="Chronological val fraction per user"),
) -> None:
    """Build a training dataset: join features + labels, exclude, split, and write X/y/splits."""
    import pandas as pd

    from taskclf.core.store import read_parquet
    from taskclf.features.build import generate_dummy_features
    from taskclf.labels.store import generate_dummy_labels
    from taskclf.train.build_dataset import build_training_dataset

    start = dt.date.fromisoformat(date_from)
    end = dt.date.fromisoformat(date_to)

    all_features: list[pd.DataFrame] = []
    all_labels: list = []
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

    manifest = build_training_dataset(
        features_df,
        all_labels,
        output_dir=Path(out_dir) / "training_dataset",
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        holdout_user_fraction=holdout_fraction,
    )

    typer.echo(f"Dataset: {manifest.total_rows} rows ({manifest.excluded_rows} excluded)")
    typer.echo(f"  Train: {manifest.train_rows}  Val: {manifest.val_rows}  Test: {manifest.test_rows}")
    typer.echo(f"  X -> {manifest.x_path}")
    typer.echo(f"  y -> {manifest.y_path}")
    typer.echo(f"  splits -> {manifest.splits_path}")


@train_app.command("lgbm")
def train_lgbm_cmd(
    date_from: str = typer.Option(..., "--from", help="Start date (YYYY-MM-DD)"),
    date_to: str = typer.Option(..., "--to", help="End date (YYYY-MM-DD, inclusive)"),
    synthetic: bool = typer.Option(False, "--synthetic", help="Generate dummy features + labels instead of reading from disk"),
    models_dir: str = typer.Option(DEFAULT_MODELS_DIR, help="Base directory for model bundles"),
    data_dir: str = typer.Option(DEFAULT_DATA_DIR, help="Processed data directory"),
    num_boost_round: int = typer.Option(DEFAULT_NUM_BOOST_ROUND, help="Number of boosting rounds"),
    class_weight: str = typer.Option("balanced", "--class-weight", help="Class-weight strategy: 'balanced' (inverse-frequency) or 'none'"),
) -> None:
    """Train a LightGBM multiclass model and save the model bundle."""
    import pandas as pd

    from taskclf.core.model_io import build_metadata, save_model_bundle
    from taskclf.core.store import read_parquet
    from taskclf.features.build import generate_dummy_features
    from taskclf.labels.projection import project_blocks_to_windows
    from taskclf.labels.store import generate_dummy_labels
    from taskclf.train.dataset import split_by_time
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

    labeled_df = project_blocks_to_windows(features_df, all_labels)
    typer.echo(f"Labeled {len(labeled_df)} / {len(features_df)} rows")

    if labeled_df.empty:
        typer.echo("No labeled rows — cannot train.", err=True)
        raise typer.Exit(code=1)

    splits = split_by_time(labeled_df)
    train_df = labeled_df.iloc[splits["train"]].reset_index(drop=True)
    val_df = labeled_df.iloc[splits["val"]].reset_index(drop=True)
    typer.echo(
        f"Train: {len(train_df)} rows, Val: {len(val_df)} rows, "
        f"Test: {len(splits['test'])} rows (held out)"
    )

    cw = class_weight if class_weight in ("balanced", "none") else "balanced"
    model, metrics, cm_df, params, cat_encoders = train_lgbm(
        train_df, val_df, num_boost_round=num_boost_round, class_weight=cw,
    )
    typer.echo(f"Macro F1: {metrics['macro_f1']}  Weighted F1: {metrics['weighted_f1']}")

    metadata = build_metadata(
        label_set=metrics["label_names"],
        train_date_from=start,
        train_date_to=end,
        params=params,
        data_provenance="synthetic" if synthetic else "real",
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


@train_app.command("evaluate")
def train_evaluate_cmd(
    model_dir: str = typer.Option(..., "--model-dir", help="Path to a model run directory"),
    date_from: str = typer.Option(..., "--from", help="Start date (YYYY-MM-DD)"),
    date_to: str = typer.Option(..., "--to", help="End date (YYYY-MM-DD, inclusive)"),
    synthetic: bool = typer.Option(False, "--synthetic", help="Generate dummy features + labels"),
    data_dir: str = typer.Option(DEFAULT_DATA_DIR, help="Processed data directory"),
    out_dir: str = typer.Option(DEFAULT_OUT_DIR, help="Output directory for evaluation artifacts"),
    holdout_fraction: float = typer.Option(0.0, "--holdout-fraction", help="Fraction of users held out for unseen-user evaluation"),
    reject_threshold: float = typer.Option(DEFAULT_REJECT_THRESHOLD, "--reject-threshold", help="Max-probability below which prediction is rejected"),
) -> None:
    """Run full evaluation of a trained model: metrics, calibration, acceptance checks."""
    import pandas as pd
    from rich.console import Console
    from rich.table import Table

    from taskclf.core.model_io import load_model_bundle
    from taskclf.core.store import read_parquet
    from taskclf.features.build import generate_dummy_features
    from taskclf.labels.projection import project_blocks_to_windows
    from taskclf.labels.store import generate_dummy_labels
    from taskclf.train.dataset import split_by_time
    from taskclf.train.evaluate import evaluate_model, write_evaluation_artifacts

    console = Console()

    model, metadata, cat_encoders = load_model_bundle(Path(model_dir))
    typer.echo(f"Loaded model from {model_dir} (schema={metadata.schema_hash})")

    start = dt.date.fromisoformat(date_from)
    end = dt.date.fromisoformat(date_to)

    all_features: list[pd.DataFrame] = []
    all_labels: list = []
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
    labeled_df = project_blocks_to_windows(features_df, all_labels)

    if labeled_df.empty:
        typer.echo("No labeled rows — cannot evaluate.", err=True)
        raise typer.Exit(code=1)

    splits = split_by_time(labeled_df, holdout_user_fraction=holdout_fraction)
    test_df = labeled_df.iloc[splits["test"]].reset_index(drop=True)
    holdout_users = splits.get("holdout_users", [])

    if test_df.empty:
        typer.echo("Test set is empty — cannot evaluate.", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"Evaluating on {len(test_df)} test rows ({len(holdout_users)} holdout users)")

    report = evaluate_model(
        model, test_df,
        cat_encoders=cat_encoders,
        holdout_users=holdout_users,
        reject_threshold=reject_threshold,
    )

    # -- Overall metrics table --
    overall = Table(title="Overall Metrics")
    overall.add_column("Metric", style="bold")
    overall.add_column("Value", justify="right")
    overall.add_row("Macro F1", f"{report.macro_f1:.4f}")
    overall.add_row("Weighted F1", f"{report.weighted_f1:.4f}")
    overall.add_row("Reject Rate", f"{report.reject_rate:.4f}")
    if report.seen_user_f1 is not None:
        overall.add_row("Seen-User F1", f"{report.seen_user_f1:.4f}")
    if report.unseen_user_f1 is not None:
        overall.add_row("Unseen-User F1", f"{report.unseen_user_f1:.4f}")
    console.print(overall)

    # -- Per-class table --
    pc_table = Table(title="Per-Class Metrics")
    pc_table.add_column("Class", style="bold")
    pc_table.add_column("Precision", justify="right")
    pc_table.add_column("Recall", justify="right")
    pc_table.add_column("F1", justify="right")
    for cls in report.label_names:
        m = report.per_class.get(cls, {})
        pc_table.add_row(
            cls,
            f"{m.get('precision', 0):.4f}",
            f"{m.get('recall', 0):.4f}",
            f"{m.get('f1', 0):.4f}",
        )
    console.print(pc_table)

    # -- Per-user table --
    pu_table = Table(title="Per-User Macro F1")
    pu_table.add_column("User", style="bold")
    pu_table.add_column("Rows", justify="right")
    pu_table.add_column("Macro F1", justify="right")
    for uid, um in report.per_user.items():
        pu_table.add_row(uid, str(int(um.get("count", 0))), f"{um.get('macro_f1', 0):.4f}")
    console.print(pu_table)

    # -- Acceptance checks table --
    acc_table = Table(title="Acceptance Checks")
    acc_table.add_column("Check", style="bold")
    acc_table.add_column("Result")
    acc_table.add_column("Detail")
    for check_name, passed in report.acceptance_checks.items():
        status = "[green]PASS[/green]" if passed else "[red]FAIL[/red]"
        acc_table.add_row(check_name, status, report.acceptance_details.get(check_name, ""))
    console.print(acc_table)

    # -- Stratification warnings --
    if report.stratification.get("warnings"):
        for w in report.stratification["warnings"]:
            console.print(f"[yellow]WARNING:[/yellow] {w}")

    # -- Write artifacts --
    out = Path(out_dir)
    paths = write_evaluation_artifacts(report, out)
    for name, p in paths.items():
        typer.echo(f"  {name}: {p}")

    all_pass = all(report.acceptance_checks.values())
    if all_pass:
        console.print("[bold green]All acceptance checks passed.[/bold green]")
    else:
        console.print("[bold red]Some acceptance checks failed.[/bold red]")


@train_app.command("tune-reject")
def train_tune_reject_cmd(
    model_dir: str = typer.Option(..., "--model-dir", help="Path to a model run directory"),
    date_from: str = typer.Option(..., "--from", help="Start date (YYYY-MM-DD)"),
    date_to: str = typer.Option(..., "--to", help="End date (YYYY-MM-DD, inclusive)"),
    synthetic: bool = typer.Option(False, "--synthetic", help="Generate dummy features + labels"),
    data_dir: str = typer.Option(DEFAULT_DATA_DIR, help="Processed data directory"),
    out_dir: str = typer.Option(DEFAULT_OUT_DIR, help="Output directory for tuning report"),
) -> None:
    """Sweep reject thresholds on a validation set and recommend the best one."""
    import json

    import pandas as pd
    from rich.console import Console
    from rich.table import Table

    from taskclf.core.model_io import load_model_bundle
    from taskclf.core.store import read_parquet
    from taskclf.features.build import generate_dummy_features
    from taskclf.labels.projection import project_blocks_to_windows
    from taskclf.labels.store import generate_dummy_labels
    from taskclf.train.dataset import split_by_time
    from taskclf.train.evaluate import tune_reject_threshold

    console = Console()

    model, metadata, cat_encoders = load_model_bundle(Path(model_dir))
    typer.echo(f"Loaded model from {model_dir} (schema={metadata.schema_hash})")

    start = dt.date.fromisoformat(date_from)
    end = dt.date.fromisoformat(date_to)

    all_features: list[pd.DataFrame] = []
    all_labels: list = []
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
    labeled_df = project_blocks_to_windows(features_df, all_labels)

    if labeled_df.empty:
        typer.echo("No labeled rows — cannot tune.", err=True)
        raise typer.Exit(code=1)

    splits = split_by_time(labeled_df)
    val_df = splits["val"]
    if val_df.empty:
        typer.echo("Validation split is empty — using full dataset.", err=True)
        val_df = labeled_df

    typer.echo(f"Tuning reject threshold on {len(val_df)} validation rows")

    result = tune_reject_threshold(model, val_df, cat_encoders=cat_encoders)

    sweep_table = Table(title="Reject Threshold Sweep")
    sweep_table.add_column("Threshold", justify="right")
    sweep_table.add_column("Accuracy", justify="right")
    sweep_table.add_column("Reject Rate", justify="right")
    sweep_table.add_column("Coverage", justify="right")
    sweep_table.add_column("Macro F1", justify="right")

    for row in result.sweep:
        marker = " *" if row["threshold"] == result.best_threshold else ""
        sweep_table.add_row(
            f"{row['threshold']:.2f}{marker}",
            f"{row['accuracy_on_accepted']:.4f}",
            f"{row['reject_rate']:.4f}",
            f"{row['coverage']:.4f}",
            f"{row['macro_f1']:.4f}",
        )

    console.print(sweep_table)
    console.print(f"\n[bold]Recommended reject threshold:[/bold] {result.best_threshold:.4f}")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / "reject_tuning.json"
    report_path.write_text(json.dumps(result.model_dump(), indent=2))
    typer.echo(f"Tuning report: {report_path}")


# -- taxonomy -----------------------------------------------------------------
taxonomy_app = typer.Typer()
app.add_typer(taxonomy_app, name="taxonomy")


@taxonomy_app.command("validate")
def taxonomy_validate_cmd(
    config: str = typer.Option(..., "--config", help="Path to a taxonomy YAML file"),
) -> None:
    """Validate a user taxonomy YAML file and report errors."""
    from pydantic import ValidationError

    from taskclf.infer.taxonomy import load_taxonomy

    config_path = Path(config)
    if not config_path.exists():
        typer.echo(f"File not found: {config_path}", err=True)
        raise typer.Exit(code=1)

    try:
        tax = load_taxonomy(config_path)
    except (ValidationError, ValueError) as exc:
        typer.echo(f"Validation failed:\n{exc}", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"Taxonomy valid: {len(tax.buckets)} buckets, version={tax.version}")
    for bucket in tax.buckets:
        typer.echo(f"  {bucket.name}: {', '.join(bucket.core_labels)}")


@taxonomy_app.command("show")
def taxonomy_show_cmd(
    config: str = typer.Option(..., "--config", help="Path to a taxonomy YAML file"),
) -> None:
    """Display a taxonomy mapping as a Rich table."""
    from pydantic import ValidationError
    from rich.console import Console
    from rich.table import Table

    from taskclf.infer.taxonomy import load_taxonomy

    console = Console()
    config_path = Path(config)
    if not config_path.exists():
        typer.echo(f"File not found: {config_path}", err=True)
        raise typer.Exit(code=1)

    try:
        tax = load_taxonomy(config_path)
    except (ValidationError, ValueError) as exc:
        typer.echo(f"Validation failed:\n{exc}", err=True)
        raise typer.Exit(code=1)

    table = Table(title=f"Taxonomy Mapping (v{tax.version})")
    table.add_column("Bucket", style="bold")
    table.add_column("Core Labels")
    table.add_column("Color")
    table.add_column("Description")

    for bucket in tax.buckets:
        table.add_row(
            bucket.name,
            ", ".join(bucket.core_labels),
            bucket.color,
            bucket.description,
        )

    console.print(table)

    info = Table(title="Advanced Settings")
    info.add_column("Setting", style="bold")
    info.add_column("Value")
    info.add_row("Aggregation", tax.advanced.probability_aggregation)
    info.add_row("Min confidence", str(tax.advanced.min_confidence_for_mapping))
    info.add_row("Reject label", tax.reject.mixed_label_name)
    if tax.user_id:
        info.add_row("User ID", tax.user_id)
    console.print(info)


@taxonomy_app.command("init")
def taxonomy_init_cmd(
    out: str = typer.Option("configs/user_taxonomy.yaml", "--out", help="Output path for the generated taxonomy YAML"),
) -> None:
    """Generate a default taxonomy YAML (identity mapping: one bucket per core label)."""
    from taskclf.infer.taxonomy import default_taxonomy, save_taxonomy

    out_path = Path(out)
    if out_path.exists():
        typer.echo(f"File already exists: {out_path}", err=True)
        raise typer.Exit(code=1)

    config = default_taxonomy()
    save_taxonomy(config, out_path)
    typer.echo(f"Default taxonomy written to {out_path} ({len(config.buckets)} buckets)")


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
    reject_threshold: float = typer.Option(DEFAULT_REJECT_THRESHOLD, "--reject-threshold", help="Max-probability below which prediction is rejected as Mixed/Unknown"),
    taxonomy_config: str | None = typer.Option(None, "--taxonomy", help="Path to a taxonomy YAML file for user-specific label mapping"),
) -> None:
    """Run batch inference: predict, smooth, and segmentize."""
    import pandas as pd

    from taskclf.core.defaults import MIXED_UNKNOWN
    from taskclf.core.metrics import reject_rate
    from taskclf.core.model_io import load_model_bundle
    from taskclf.core.store import read_parquet
    from taskclf.features.build import generate_dummy_features
    from taskclf.infer.batch import (
        run_batch_inference,
        write_predictions_csv,
        write_segments_json,
    )

    taxonomy = None
    if taxonomy_config is not None:
        from taskclf.infer.taxonomy import load_taxonomy

        taxonomy = load_taxonomy(Path(taxonomy_config))
        typer.echo(f"Loaded taxonomy from {taxonomy_config}")

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

    result = run_batch_inference(
        model, features_df,
        cat_encoders=cat_encoders,
        smooth_window=smooth_window,
        reject_threshold=reject_threshold,
        taxonomy=taxonomy,
    )
    rr = reject_rate(result.smoothed_labels, MIXED_UNKNOWN)
    typer.echo(
        f"Predicted {len(result.smoothed_labels)} buckets -> {len(result.segments)} segments "
        f"(reject rate: {rr:.1%})"
    )

    out = Path(out_dir)
    pred_path = write_predictions_csv(
        features_df, result.smoothed_labels, out / "predictions.csv",
        confidences=result.confidences, is_rejected=result.is_rejected,
        mapped_labels=result.mapped_labels,
        core_probs=result.core_probs,
    )
    seg_path = write_segments_json(result.segments, out / "segments.json")

    typer.echo(f"Predictions: {pred_path}")
    typer.echo(f"Segments:    {seg_path}")
    if result.mapped_labels is not None:
        typer.echo("Taxonomy mapping applied: mapped_label column included in predictions")


@infer_app.command("online")
def infer_online_cmd(
    model_dir: str = typer.Option(..., "--model-dir", help="Path to a model run directory"),
    poll_seconds: int = typer.Option(DEFAULT_POLL_SECONDS, "--poll-seconds", help="Seconds between polling iterations"),
    aw_host: str = typer.Option(DEFAULT_AW_HOST, "--aw-host", help="ActivityWatch server URL"),
    smooth_window: int = typer.Option(DEFAULT_SMOOTH_WINDOW, "--smooth-window", help="Rolling majority smoothing window size"),
    title_salt: str = typer.Option(DEFAULT_TITLE_SALT, "--title-salt", help="Salt for hashing window titles"),
    out_dir: str = typer.Option(DEFAULT_OUT_DIR, help="Output directory for predictions and segments"),
    reject_threshold: float = typer.Option(DEFAULT_REJECT_THRESHOLD, "--reject-threshold", help="Max-probability below which prediction is rejected as Mixed/Unknown"),
    taxonomy_config: str | None = typer.Option(None, "--taxonomy", help="Path to a taxonomy YAML file for user-specific label mapping"),
    calibrator_config: str | None = typer.Option(None, "--calibrator", help="Path to a calibrator JSON file for probability calibration"),
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
        reject_threshold=reject_threshold,
        taxonomy_path=Path(taxonomy_config) if taxonomy_config else None,
        calibrator_path=Path(calibrator_config) if calibrator_config else None,
    )


@infer_app.command("baseline")
def infer_baseline_cmd(
    date_from: str = typer.Option(..., "--from", help="Start date (YYYY-MM-DD)"),
    date_to: str = typer.Option(..., "--to", help="End date (YYYY-MM-DD, inclusive)"),
    synthetic: bool = typer.Option(False, "--synthetic", help="Generate dummy features instead of reading from disk"),
    data_dir: str = typer.Option(DEFAULT_DATA_DIR, help="Processed data directory"),
    out_dir: str = typer.Option(DEFAULT_OUT_DIR, help="Output directory for predictions and segments"),
    smooth_window: int = typer.Option(DEFAULT_SMOOTH_WINDOW, help="Rolling majority smoothing window size"),
) -> None:
    """Run rule-based baseline inference (no ML model required)."""
    import pandas as pd

    from taskclf.core.defaults import MIXED_UNKNOWN
    from taskclf.core.store import read_parquet
    from taskclf.features.build import generate_dummy_features
    from taskclf.infer.baseline import run_baseline_inference
    from taskclf.infer.batch import write_predictions_csv, write_segments_json

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

    smoothed_labels, segments = run_baseline_inference(
        features_df, smooth_window=smooth_window,
    )

    from taskclf.core.metrics import reject_rate

    rr = reject_rate(smoothed_labels, MIXED_UNKNOWN)
    typer.echo(
        f"Baseline predicted {len(smoothed_labels)} buckets -> "
        f"{len(segments)} segments (reject rate: {rr:.1%})"
    )

    out = Path(out_dir)
    pred_path = write_predictions_csv(features_df, smoothed_labels, out / "baseline_predictions.csv")
    seg_path = write_segments_json(segments, out / "baseline_segments.json")

    typer.echo(f"Predictions: {pred_path}")
    typer.echo(f"Segments:    {seg_path}")


@infer_app.command("compare")
def infer_compare_cmd(
    model_dir: str = typer.Option(..., "--model-dir", help="Path to a model run directory"),
    date_from: str = typer.Option(..., "--from", help="Start date (YYYY-MM-DD)"),
    date_to: str = typer.Option(..., "--to", help="End date (YYYY-MM-DD, inclusive)"),
    synthetic: bool = typer.Option(False, "--synthetic", help="Generate dummy features + labels"),
    data_dir: str = typer.Option(DEFAULT_DATA_DIR, help="Processed data directory"),
    out_dir: str = typer.Option(DEFAULT_OUT_DIR, help="Output directory for comparison report"),
) -> None:
    """Compare rule baseline vs ML model on labeled data."""
    import json

    import pandas as pd
    from rich.console import Console
    from rich.table import Table

    from taskclf.core.defaults import MIXED_UNKNOWN
    from taskclf.core.metrics import compare_baselines
    from taskclf.core.model_io import load_model_bundle
    from taskclf.core.store import read_parquet
    from taskclf.core.types import LABEL_SET_V1
    from taskclf.features.build import generate_dummy_features
    from taskclf.infer.baseline import predict_baseline
    from taskclf.infer.batch import predict_labels
    from taskclf.labels.projection import project_blocks_to_windows
    from taskclf.labels.store import generate_dummy_labels

    console = Console()
    start = dt.date.fromisoformat(date_from)
    end = dt.date.fromisoformat(date_to)

    all_features: list[pd.DataFrame] = []
    all_labels: list = []
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
    labeled_df = project_blocks_to_windows(features_df, all_labels)

    if labeled_df.empty:
        typer.echo("No labeled rows — cannot compare.", err=True)
        raise typer.Exit(code=1)

    y_true = list(labeled_df["label"].values)
    typer.echo(f"Comparing on {len(y_true)} labeled windows")

    baseline_preds = predict_baseline(labeled_df)

    from sklearn.preprocessing import LabelEncoder

    model, _meta, cat_encoders = load_model_bundle(Path(model_dir))
    le = LabelEncoder()
    le.fit(sorted(LABEL_SET_V1))
    model_preds = predict_labels(model, labeled_df, le, cat_encoders=cat_encoders)

    label_names = sorted(LABEL_SET_V1)
    results = compare_baselines(
        y_true,
        {"baseline": baseline_preds, "model": model_preds},
        label_names,
        reject_label=MIXED_UNKNOWN,
    )

    summary_table = Table(title="Baseline vs Model Comparison")
    summary_table.add_column("Metric", style="bold")
    summary_table.add_column("Baseline", justify="right")
    summary_table.add_column("Model", justify="right")
    summary_table.add_column("Delta", justify="right")

    bl = results["baseline"]
    ml = results["model"]
    for metric in ("macro_f1", "weighted_f1", "reject_rate"):
        delta = ml[metric] - bl[metric]
        sign = "+" if delta >= 0 else ""
        summary_table.add_row(
            metric, f"{bl[metric]:.4f}", f"{ml[metric]:.4f}", f"{sign}{delta:.4f}",
        )
    console.print(summary_table)

    detail_table = Table(title="Per-Class F1")
    detail_table.add_column("Class", style="bold")
    detail_table.add_column("Baseline F1", justify="right")
    detail_table.add_column("Model F1", justify="right")

    all_labels_list = bl.get("label_names", label_names)
    for cls in all_labels_list:
        bl_f1 = bl["per_class"].get(cls, {}).get("f1", 0.0)
        ml_f1 = ml["per_class"].get(cls, {}).get("f1", 0.0)
        detail_table.add_row(cls, f"{bl_f1:.4f}", f"{ml_f1:.4f}")
    console.print(detail_table)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / "baseline_vs_model.json"
    report_path.write_text(json.dumps(results, indent=2))
    typer.echo(f"Full comparison report: {report_path}")


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
