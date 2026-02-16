# FeatureSchema v1, validation, versioning

# TODO: Invariants to enforce in `core/schema.py`

# * Feature rows are **bucketed** (e.g., 60s) and have `bucket_start_ts`.
# * Every feature row carries:

#   * `schema_version`
#   * `schema_hash`
#   * `source_ids` (which collectors produced it)
# * No raw keystrokes, no raw window titles (enforce via validation).
