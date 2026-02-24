# train.dataset

Join features with label spans and split by time.

## Functions

### assign_labels_to_buckets

Joins feature rows with label spans. Each feature row is assigned the
label of the first span whose `[start_ts, end_ts)` interval covers the
row's `bucket_start_ts`. Rows with no covering span are dropped.

### split_by_day

Two-way split (train / val) by calendar day. The last unique day
becomes validation. Falls back to 80/20 chronological split if only
one day is present.

### split_by_time

Three-way chronological split (train / val / test) with optional
cross-user holdout. For each non-holdout user, rows are sorted by
`bucket_start_ts` and split at `train_ratio` / `val_ratio` /
remainder boundaries. Holdout users have all data placed in the test
set only.

Returns a dict with `"train"`, `"val"`, `"test"` (index lists) and
`"holdout_users"`.

::: taskclf.train.dataset
