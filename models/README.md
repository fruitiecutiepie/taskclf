# models/

Trained model bundles. Each run is self-contained and should be reproducible.

## Layout
`YYYY-MM-DD_HHMMSS_run-XXXX/`
- `model.txt` or `model.bin`
- `metadata.json` (schema hash, label set, train range, params, git commit)
- `metrics.json`
- `confusion_matrix.csv`
- optional calibration artifacts

## Invariants
- Never overwrite a run directory. Create a new run.
- Inference must validate `schema_hash` matches the runtime feature schema.
