# core.types

Pydantic models for the core data contracts.

## TitlePolicy

`TitlePolicy` controls whether raw window titles may appear in a
`FeatureRow`.

| Member | Value | Behaviour |
|---|---|---|
| `HASH_ONLY` | `"hash_only"` | Default. All `raw_*` fields are rejected. |
| `RAW_WINDOW_TITLE_OPT_IN` | `"raw_window_title_opt_in"` | `raw_window_title` is accepted but excluded from `model_dump()`, preventing leakage into `data/processed/`. All other `raw_*` fields remain prohibited. |

Pass the policy via Pydantic validation context:

```python
from taskclf.core.types import FeatureRow, TitlePolicy

row = FeatureRow.model_validate(
    data,
    context={"title_policy": TitlePolicy.RAW_WINDOW_TITLE_OPT_IN},
)
row.raw_window_title   # available on the instance
row.model_dump()       # raw_window_title is NOT included
```

::: taskclf.core.types
