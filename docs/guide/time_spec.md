# Time & Windowing Specification v1

Version: 1.0  
Status: Stable  
Last Updated: 2026-02-23  

This document defines the canonical time semantics used across:

- Feature extraction
- Dataset construction
- Label projection
- Model training
- Inference
- Aggregation
- Time tracking summaries

All components MUST adhere to this specification.

---

# 1. Time Standard

## 1.1 Timestamp Format

All timestamps MUST be:

- UTC
- ISO 8601 format
- Stored as:
  - `bucket_start_ts` (inclusive)
  - `bucket_end_ts` (exclusive)

Example:

```

2026-02-23T05:30:00Z

```

Local time (for display only) must be derived from UTC using stored user timezone.

---

# 2. Window Definition

## 2.1 Fixed Window Size

Window duration:

```

WINDOW_SIZE_SECONDS = 60

```

All feature aggregation operates over fixed, non-overlapping windows.

Windows are aligned to wall clock boundaries.

Example:

05:30:00–05:31:00  
05:31:00–05:32:00  

Rolling windows are NOT allowed.

---

## 2.2 Window Identity

Each window is uniquely identified by:

- `user_id`
- `bucket_start_ts`

Optional:
- `device_id`

---

# 3. Session Definition

A session is a contiguous period of activity separated by idle gaps.

## 3.1 Idle Threshold

```

IDLE_GAP_THRESHOLD_SECONDS = 300  (5 minutes)

```

If no interaction events are recorded for 5 minutes or more:

- The previous session ends
- A new session begins upon next activity

---

## 3.2 session_length_so_far

Definition:

Minutes since session start.

For each window:

```

session_length_so_far =
(bucket_start_ts - session_start_ts) / 60

```

If a session restarts, this resets to 0.

---

# 4. Idle Detection

Idle detection is based on:

- No keyboard events
- No mouse movement
- No clicks
- No scroll
- No app switches

If a window contains no interaction events:

- It is marked as idle candidate
- If idle gap >= IDLE_GAP_THRESHOLD_SECONDS → label becomes BreakIdle

BreakIdle overrides model prediction.

---

# 5. Partial Windows

If activity starts mid-window:

- Window still exists
- Features computed from available activity
- Missing interaction treated as zero

No partial window truncation allowed.

---

# 6. Label Projection (Block → Window)

Manual labeling is done in time blocks.

Projection rules:

1. A window is labeled if:
   - Its entire duration falls within a labeled block

2. If a window overlaps multiple labeled blocks:
   - Drop from training set

3. If a window partially overlaps a block:
   - Drop from training set

4. Unlabeled windows:
   - Not used in supervised training

---

# 7. Aggregation Rules (Time Tracking)

## 7.1 Window → Block Merge

Adjacent windows with identical final label (post-smoothing) must be merged.

## 7.2 Minimum Block Duration

```

MIN_BLOCK_DURATION_SECONDS = 180  (3 minutes)

```

If a predicted label change lasts less than 3 minutes:

- It may be smoothed to surrounding label
- Exact smoothing strategy defined in inference layer

---

# 8. Day Boundaries

Daily summaries use user local time.

Day boundary:

```

00:00:00 local time

```

Windows are grouped by local date after timezone conversion.

---

# 9. Feature Time Semantics

The following features depend on window start time:

- `hour_of_day`
- `day_of_week`

They are computed from:

```

bucket_start_ts (converted to user local timezone)

```

---

# 10. Determinism Requirements

Given identical raw event logs:

- Windowing MUST produce identical buckets
- Session segmentation MUST be deterministic
- Feature values MUST be reproducible

No randomness allowed in feature pipeline.

---

# 11. Versioning

Changing any of the following requires version bump:

- Window size
- Idle threshold
- Session definition
- Block projection policy
- Minimum block duration
