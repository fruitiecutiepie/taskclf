# infer.smooth

Post-prediction smoothing and segmentization into contiguous spans.

## Segment

Frozen dataclass representing a contiguous run of identical predicted
labels.

| Field | Type | Description |
|-------|------|-------------|
| `start_ts` | `datetime` | Start of the first bucket in the segment |
| `end_ts` | `datetime` | End of the last bucket (start + bucket width) |
| `label` | `str` | Predicted label for the entire segment |
| `bucket_count` | `int` | Number of time buckets in the segment |

## rolling_majority

Centred sliding-window majority vote smoother.

For each position, the most common label within the window wins.
Ties are broken by keeping the **original label**, which biases toward
stability.  The window is centred: for `window=5`, positions
`[i-2, i+2]` are considered (clamped at boundaries).

```python
from taskclf.infer.smooth import rolling_majority

raw = ["Coding", "Coding", "Meeting", "Coding", "Coding"]
smoothed = rolling_majority(raw, window=3)
# smoothed == ["Coding", "Coding", "Coding", "Coding", "Coding"]
```

The default window size comes from `DEFAULT_SMOOTH_WINDOW` in
[`core.defaults`](../core/defaults.md).

## segmentize

Merges consecutive identical labels into `Segment` spans.  Every input
bucket is covered by exactly one segment; segments are ordered and
non-overlapping.

```python
from taskclf.infer.smooth import segmentize

segments = segmentize(bucket_starts, smoothed_labels, bucket_seconds=60)
```

Raises `ValueError` if `bucket_starts` and `labels` differ in length
or are empty.

## flap_rate

Computes the label flap rate: `label_changes / total_windows`.

As defined in the [acceptance guide, section 5](../../guide/acceptance.md):

| Metric | Threshold |
|--------|-----------|
| Raw flap rate | &le; 0.25 |
| Smoothed flap rate | &le; 0.15 |

Returns 0.0 for sequences of length &le; 1.

## merge_short_segments

Hysteresis rule that absorbs segments shorter than
`MIN_BLOCK_DURATION_SECONDS` (default 180 s / 3 minutes) into their
neighbours.

Strategy for each short segment:

1. If either neighbour has the **same label**, merge into that
   neighbour.
2. Otherwise merge into the **longer** neighbour.
3. On a tie, prefer the **preceding** neighbour.
4. First and last segments are never removed (but may absorb
   neighbours).

The function iterates until no more short segments can be merged,
guaranteeing convergence because total segment count strictly decreases
each pass.

```python
from taskclf.infer.smooth import merge_short_segments

merged = merge_short_segments(segments, min_duration_seconds=180)
```

::: taskclf.infer.smooth
