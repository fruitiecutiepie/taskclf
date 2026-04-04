# core.store

Parquet (and future duckdb) IO primitives.

`pandas` is imported inside `read_parquet()` and `write_parquet()` so importing
`taskclf.core.store` does not eagerly load the full dataframe stack; callers
that only need other modules avoid that cost until parquet I/O runs.

::: taskclf.core.store
