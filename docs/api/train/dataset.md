# train.dataset

Time-based dataset splitting utilities.

## Functions

### split_by_time

Three-way chronological split (train / val / test) with optional
cross-user holdout. For each non-holdout user, rows are sorted by
`bucket_start_ts` and split at `train_ratio` / `val_ratio` /
remainder boundaries. Holdout users have all data placed in the test
set only.

Returns a dict with `"train"`, `"val"`, `"test"` (index lists) and
`"holdout_users"`.

::: taskclf.train.dataset
