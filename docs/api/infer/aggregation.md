# infer.aggregation

Interval-aware aggregation strategies for multi-bucket predictions.

When a tray suggestion covers a time interval containing multiple 60-second
buckets, each bucket produces an independent `WindowPrediction`. This module
reduces those per-bucket predictions into a single `(label, confidence)` pair.

## Strategies

| Strategy | Function | Behavior |
|---|---|---|
| `"majority"` | `majority_vote` | Most frequent label wins. Ties broken by first occurrence. |
| `"confidence_weighted"` | `confidence_weighted_vote` | Each bucket's confidence is accumulated per label; the label with the highest total weight wins. |
| `"highest_probability"` | `highest_total_probability` | Per-class probability columns are summed across buckets; the class with the highest total wins. |

## Entry point

```python
from taskclf.infer.aggregation import aggregate_interval

label, confidence = aggregate_interval(predictions, strategy="majority")
```

`aggregate_interval` dispatches to the strategy function and returns the
winning label together with the mean confidence of buckets that voted for it.

## API

::: taskclf.infer.aggregation
