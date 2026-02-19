"""Typer CLI entrypoint and command definitions for taskclf."""

# TODO: remaining commands:
# * taskclf ingest aw --input <path> --out data/raw/aw/YYYY-MM-DD/
# * taskclf labels import --file labels.csv
# * taskclf infer online --poll-seconds 60
# * taskclf report daily --date 2026-02-13

# TODO: schema hash gate — at inference time, refuse to run if mismatch.

import datetime as dt
from pathlib import Path

import typer

app = typer.Typer()

# -- features ----------------------------------------------------------------
features_app = typer.Typer()
app.add_typer(features_app, name="features")


@features_app.command("build")
def features_build(date: str = typer.Option(..., help="Date in YYYY-MM-DD format")) -> None:
    """Build per-minute feature rows for a given date and write to parquet."""
    from taskclf.features.build import build_features_for_date

    parsed_date = dt.date.fromisoformat(date)
    out_path = build_features_for_date(parsed_date, Path("data/processed"))
    typer.echo(f"Wrote features to {out_path}")


# -- train --------------------------------------------------------------------
train_app = typer.Typer()
app.add_typer(train_app, name="train")


@train_app.command("lgbm")
def train_lgbm_cmd(
    date_from: str = typer.Option(..., "--from", help="Start date (YYYY-MM-DD)"),
    date_to: str = typer.Option(..., "--to", help="End date (YYYY-MM-DD, inclusive)"),
    synthetic: bool = typer.Option(False, "--synthetic", help="Generate dummy features + labels instead of reading from disk"),
    models_dir: str = typer.Option("models", help="Base directory for model bundles"),
    data_dir: str = typer.Option("data/processed", help="Processed data directory"),
    num_boost_round: int = typer.Option(100, help="Number of boosting rounds"),
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

    model, metrics, cm_df, params = train_lgbm(
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
    )
    typer.echo(f"Model bundle saved to {run_dir}")


if __name__ == "__main__":
    app()
