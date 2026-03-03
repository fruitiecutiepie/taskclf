# labels.weak_rules

Heuristic weak-labeling rules that map feature rows to task-type labels.

## Overview

Weak rules provide an automated, low-confidence alternative to manual
labeling.  Each rule inspects a single feature column and, when matched,
proposes a [`LabelSpan`](../core/types.md) with
`provenance="weak:<rule_name>"`.  Rules are evaluated in list order;
the **first match wins**.

Weak labels share the same `LabelSpan` structure as gold labels.
The `provenance` field distinguishes them — gold labels use `"manual"`,
while weak labels use the `"weak:<rule_name>"` convention.  This allows
downstream consumers (training, projection) to filter or weight labels
by origin.

## WeakRule

Frozen dataclass representing a single heuristic rule.

| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` | Human-readable identifier, also used in `provenance` |
| `field` | `str` | Feature column to inspect (e.g. `"app_id"`) |
| `pattern` | `str` | Value the column must equal for the rule to fire |
| `label` | `str` | Task-type label to assign (must be in `LABEL_SET_V1`) |
| `confidence` | `float \| None` | Optional confidence attached to produced spans |

Construction raises `ValueError` if `label` is not in `LABEL_SET_V1`.

## Built-in rule maps

The module provides three dictionaries that feed `build_default_rules()`.
They are ordered by specificity when assembled into the default list.

### `APP_ID_RULES` (highest priority)

Maps reverse-domain `app_id` values to labels.

| `app_id` | Label |
|----------|-------|
| `com.apple.Terminal` | Build |
| `com.microsoft.VSCode` | Build |
| `com.jetbrains.intellij` | Build |
| `com.googlecode.iterm2` | Build |
| `org.mozilla.firefox` | ReadResearch |
| `com.google.Chrome` | ReadResearch |
| `com.apple.Safari` | ReadResearch |
| `com.apple.mail` | Communicate |
| `com.tinyspeck.slackmacgap` | Communicate |
| `us.zoom.xos` | Meet |
| `com.apple.Notes` | Write |
| `com.apple.finder` | BreakIdle |

### `APP_CATEGORY_RULES`

Maps `app_category` values to labels.

| Category | Label |
|----------|-------|
| `meeting` | Meet |
| `chat` | Communicate |
| `email` | Communicate |
| `editor` | Build |
| `terminal` | Build |
| `devtools` | Debug |
| `docs` | Write |
| `design` | Write |
| `media` | BreakIdle |
| `file_manager` | BreakIdle |

### `DOMAIN_CATEGORY_RULES` (lowest priority)

Maps browser `domain_category` values to labels.

| Domain | Label |
|--------|-------|
| `code_hosting` | Build |
| `email_web` | Communicate |
| `chat` | Communicate |
| `social` | BreakIdle |
| `video` | BreakIdle |
| `news` | ReadResearch |
| `docs` | ReadResearch |
| `search` | ReadResearch |
| `productivity` | Write |

## Functions

### `build_default_rules`

```python
def build_default_rules() -> list[WeakRule]
```

Builds the default rule list from the three built-in maps, ordered by
specificity: `APP_ID_RULES` first, then `APP_CATEGORY_RULES`, then
`DOMAIN_CATEGORY_RULES`.

### `match_rule`

```python
def match_rule(
    row: dict[str, Any],
    rules: Sequence[WeakRule],
) -> tuple[str, str] | None
```

Matches a single feature row (as a dict) against an ordered list of
rules.  Returns `(label, rule_name)` on the first match, or `None`
if no rule fires.

### `apply_weak_rules`

```python
def apply_weak_rules(
    features_df: pd.DataFrame,
    rules: Sequence[WeakRule] | None = None,
    user_id: str | None = None,
    bucket_seconds: int = DEFAULT_BUCKET_SECONDS,
) -> list[LabelSpan]
```

Applies rules to every row in *features_df*.  Consecutive buckets
(by `bucket_start_ts`) that receive the **same label** are merged
into a single `LabelSpan`.  A new span starts when the label changes
or there is a time gap between buckets.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `features_df` | — | DataFrame with `bucket_start_ts`, `bucket_end_ts`, and feature columns |
| `rules` | `None` | Rule list; defaults to `build_default_rules()` |
| `user_id` | `None` | User id attached to every produced span |
| `bucket_seconds` | `60` | Expected bucket duration for gap detection |

## Usage

```python
import pandas as pd
from taskclf.labels.weak_rules import apply_weak_rules, build_default_rules

rules = build_default_rules()
weak_spans = apply_weak_rules(features_df, rules=rules, user_id="u1")

for span in weak_spans:
    print(f"{span.start_ts} → {span.end_ts}  {span.label}  ({span.provenance})")
```

Custom rules can be mixed with or replace the defaults:

```python
from taskclf.labels.weak_rules import WeakRule, apply_weak_rules

custom_rules = [
    WeakRule(name="figma", field="app_id", pattern="com.figma.Desktop", label="Write"),
]
spans = apply_weak_rules(features_df, rules=custom_rules)
```

## See also

- [`labels.store`](store.md) — persisting label spans
- [`labels.projection`](projection.md) — projecting spans onto feature windows
- [`core.types.LabelSpan`](../core/types.md) — the shared label data model

::: taskclf.labels.weak_rules
