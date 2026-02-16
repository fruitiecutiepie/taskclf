# Typer app: commands call pipelines

# TODO: Implement in `taskclf.cli.main`:

# * `taskclf ingest aw --input <path> --out data/raw/aw/YYYY-MM-DD/`
# * `taskclf features build --date 2026-02-13`
# * `taskclf labels import --file labels.csv`
# * `taskclf train lgbm --from 2026-02-01 --to 2026-02-13`
# * `taskclf infer online --poll-seconds 60`
# * `taskclf report daily --date 2026-02-13`

# 1. `ingest` pulls/reads ActivityWatch export
# 2. `features` builds per-minute rows → parquet partitioned by day
# 3. `labels` you create/merge label spans
# 4. `train` builds dataset, splits by day, trains model, writes `models/<run_id>/`
# 5. `report` generates daily summaries

# This is your “capabilities” layer; everything else is internal.

# TODO: “Schema hash” gate

# At training time, compute hash of:

# * ordered feature column names + dtypes
# * label set
# * preprocessing options
#   Store it. At inference time, refuse to run if mismatch.

# This prevents silent skew when you add/remove features.

import typer
app = typer.Typer()

@app.command()
def features_build(date: str):
    from taskclf.train import dataset  # or pipeline module
    # call pipeline function; keep CLI thin

if __name__ == "__main__":
    app()
