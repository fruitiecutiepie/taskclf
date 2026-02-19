# TODO: remaining commands:
# * taskclf ingest aw --input <path> --out data/raw/aw/YYYY-MM-DD/
# * taskclf labels import --file labels.csv
# * taskclf train lgbm --from 2026-02-01 --to 2026-02-13
# * taskclf infer online --poll-seconds 60
# * taskclf report daily --date 2026-02-13

# TODO: schema hash gate — at training time, compute and store hash of
# ordered feature column names + dtypes, label set, preprocessing options.
# At inference time, refuse to run if mismatch.

import datetime as dt
from pathlib import Path

import typer

app = typer.Typer()
features_app = typer.Typer()
app.add_typer(features_app, name="features")


@features_app.command("build")
def features_build(date: str = typer.Option(..., help="Date in YYYY-MM-DD format")) -> None:
    """Build per-minute feature rows for a given date and write to parquet."""
    from taskclf.features.build import build_features_for_date

    parsed_date = dt.date.fromisoformat(date)
    out_path = build_features_for_date(parsed_date, Path("data/processed"))
    typer.echo(f"Wrote features to {out_path}")


if __name__ == "__main__":
    app()
