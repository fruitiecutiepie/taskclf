## Data contracts (primitives vs structure)

### Primitives (immutable-ish “world state”)

* `data/raw/` : ActivityWatch export snapshots (or JSONL)
* `data/processed/features_v1/` : partitioned parquet by date
* `data/processed/labels_v1/` : label spans by date
* `models/` : model artifacts with metadata

### Structures (pipelines)

* ETL pipeline reads raw → produces features parquet
* Training pipeline reads features + labels → produces model
* Inference pipeline reads new events → emits predictions + segments
