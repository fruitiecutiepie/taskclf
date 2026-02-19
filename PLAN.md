We’ll optimize for:

* Fast feedback loop
* Privacy-safe logging
* Strong baseline in < 1 week
* Architecture that scales

---

# Phase 0 — Define the Target (1 hour)

### 1️⃣ Lock the label schema (don’t overcomplicate)

Start with 6 classes:

```
coding
writing_docs
messaging_email
browsing_research
meetings_calls
break_idle
```

Write this down in a `labels.yaml`.
Don’t change it for 2 weeks.

---

# Phase 1 — Data Collection (Day 1)

## Step 1: Install ActivityWatch

It already logs:

* Active app
* Window title
* Idle time

Install:
[https://github.com/ActivityWatch/activitywatch](https://github.com/ActivityWatch/activitywatch)

Let it run in the background.

---

## Step 2: Create a feature extraction script

Create repo:

```
task-classifier/
  data/
  notebooks/
  src/
  models/
```

Inside `src/feature_pipeline.py`:

Extract per-minute rows with:

### Required Features

**Context**

* app_id
* window_title_hash
* is_browser
* is_editor
* is_terminal
* app_switch_count_last_5m

**Keyboard (if you add logger later)**

* keys_per_min
* backspace_ratio
* shortcut_rate

**Mouse**

* clicks_per_min
* scroll_events_per_min
* mouse_distance

**Temporal**

* hour_of_day
* day_of_week
* session_length_so_far

If you want quick start:
Begin with app + time only.
That alone will give decent performance.

Export to:

```
data/features.parquet
```

---

# Phase 2 — Labeling (Days 2–5)

## Step 3: Build minimal labeling UI (simple > perfect)

Option A (fastest):

* Export top continuous segments by app
* Label in CSV

Option B (better):
Small Streamlit app:

```
streamlit run label_app.py
```

Shows:

* app
* window title
* duration
* time range

You click a label.

### Label only:

* Long segments (>5 min)
* Most frequent apps

Target:
~6–10 hours labeled total time

---

# Phase 3 — Baseline Model (Day 5–6)

## Step 4: Train baseline

Start simple:

```python
import lightgbm as lgb
```

Use:

* Multiclass objective
* 100–300 trees
* No heavy tuning

Important:
Split by **day**, not random rows.

Metrics:

* Macro F1
* Confusion matrix

If F1 > 0.7 after 1 week of data → you’re doing great.

---

# Phase 4 — Improve Signal (Week 2)

Add:

### 1️⃣ App Category Mapping Layer

Manual mapping:

```
VSCode → coding
Chrome (docs.google.com) → writing_docs
Slack → messaging_email
```

Treat as feature, not label override.

### 2️⃣ Rolling Window Smoothing

Prediction smoothing:

```
final_label[t] = majority(pred[t-2:t+2])
```

This massively stabilizes output.

---

# Phase 5 — Inference Loop

## Step 5: Real-time inference service

Simple loop:

```
every 60s:
  extract last-minute features
  predict
  append to daily log
```

Export daily summary:

```
coding: 4h 12m
writing: 1h 38m
```

Now you have:
Auto time tracking.

---

# Architecture (Important for You)

Keep strict boundaries:

```
collector → feature extractor → model → inference → report
```

Never mix them.

* Collector = platform dependent
* Features = stable schema
* Model = swappable
* Inference = stateless consumer

Design feature schema as a versioned contract:

```
feature_schema_v1.json
```

This prevents silent breakage later.

---

## Execution architecture (how it runs)

You want two “modes”:

### A) Batch mode (repeatable, easiest to debug)

1. `ingest` pulls/reads ActivityWatch export
2. `features` builds per-minute rows → parquet partitioned by day
3. `labels` you create/merge label spans
4. `train` builds dataset, splits by day, trains model, writes `models/<run_id>/`
5. `report` generates daily summaries

### B) Online mode (real-time predictions)

A lightweight loop:

* every 60s:

  * read last 1–5 minutes of events
  * build features for latest bucket
  * predict + smooth
  * append to `artifacts/predictions/YYYY-MM-DD.parquet`
* end of day:

  * produce summary report

Online mode should **not** retrain. Retraining is batch-only.

---

# Privacy Rules (Non-negotiable)

Do NOT store:

* Raw keystrokes
* Full window titles (store hash or tokenize locally)
* Clipboard

Only store:
Aggregated counts per minute.

---

# Expected Timeline

Day 1: Data collecting
Day 3: First labeled dataset
Day 6: First model
Week 2: Stable auto-tracking

---

# When You’ll Know It’s Working

You stop manually tracking time.

That’s the success metric.
