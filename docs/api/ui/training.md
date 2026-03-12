# ui.server — Training Endpoints

REST endpoints for triggering model training, feature building, and
model management from the web UI.

## Endpoints

### `POST /api/train/start`

Start a LightGBM training job in the background. Only one job runs at
a time; returns 409 if a job is already running.

**Request body:**

| Field | Type | Default | Description |
|---|---|---|---|
| `date_from` | `str` | *(required)* | Start date (YYYY-MM-DD) |
| `date_to` | `str` | *(required)* | End date (YYYY-MM-DD, inclusive) |
| `num_boost_round` | `int` | `100` | Number of boosting rounds |
| `class_weight` | `str` | `"balanced"` | `"balanced"` or `"none"` |
| `synthetic` | `bool` | `false` | Use dummy features and labels |

**Response (202):**

```json
{
  "job_id": "a1b2c3d4e5f6",
  "status": "running",
  "step": "initializing",
  "progress_pct": 0,
  "message": "Starting…",
  "error": null,
  "metrics": null,
  "model_dir": null,
  "started_at": "2026-03-12T10:00:00+00:00",
  "finished_at": null
}
```

### `POST /api/train/build-features`

Build features for a date range in the background. Runs
`build_features_for_date()` for each date.

**Request body:**

| Field | Type | Description |
|---|---|---|
| `date_from` | `str` | Start date (YYYY-MM-DD) |
| `date_to` | `str` | End date (YYYY-MM-DD, inclusive) |

**Response (202):** Same schema as `/api/train/start`.

### `GET /api/train/status`

Poll the current training job state. Returns the last job's status
even after completion.

**Response:**

```json
{
  "job_id": "a1b2c3d4e5f6",
  "status": "complete",
  "step": "done",
  "progress_pct": 100,
  "message": "Model saved to 2026-03-12_100000_run-0042",
  "error": null,
  "metrics": {"macro_f1": 0.85, "weighted_f1": 0.87},
  "model_dir": "/path/to/models/2026-03-12_100000_run-0042",
  "started_at": "2026-03-12T10:00:00+00:00",
  "finished_at": "2026-03-12T10:01:30+00:00"
}
```

Status values: `"idle"`, `"running"`, `"complete"`, `"failed"`.

### `POST /api/train/cancel`

Best-effort cancel of the running training job. Returns 409 if no job
is running.

### `GET /api/train/models`

List available model bundles from the models directory.

**Response:**

```json
[
  {
    "model_id": "2026-03-12_100000_run-0042",
    "path": "/path/to/models/2026-03-12_100000_run-0042",
    "valid": true,
    "invalid_reason": null,
    "macro_f1": 0.85,
    "weighted_f1": 0.87,
    "created_at": "2026-03-12T10:01:30"
  }
]
```

### `GET /api/train/data-check`

Check data readiness for a date range.

**Query parameters:**

| Param | Type | Description |
|---|---|---|
| `date_from` | `str` | Start date (YYYY-MM-DD) |
| `date_to` | `str` | End date (YYYY-MM-DD, inclusive) |

**Response:**

```json
{
  "date_from": "2026-02-01",
  "date_to": "2026-03-01",
  "dates_with_features": ["2026-02-01", "2026-02-02"],
  "dates_missing_features": ["2026-02-03"],
  "total_feature_rows": 1440,
  "label_span_count": 25
}
```

## WebSocket Events

Training progress is streamed to connected WebSocket clients
(`/ws/predictions`) via the shared EventBus.

### `train_progress`

Emitted at each pipeline step during training.

```json
{
  "type": "train_progress",
  "job_id": "a1b2c3d4e5f6",
  "step": "training",
  "progress_pct": 50,
  "message": "Training LightGBM (100 rounds, 500 train / 100 val)…"
}
```

Steps: `initializing`, `loading_features`, `projecting_labels`,
`splitting`, `training`, `saving`, `done`, `building_features`.

### `train_complete`

Emitted when training finishes successfully.

```json
{
  "type": "train_complete",
  "job_id": "a1b2c3d4e5f6",
  "metrics": {"macro_f1": 0.85, "weighted_f1": 0.87},
  "model_dir": "/path/to/models/2026-03-12_100000_run-0042"
}
```

### `train_failed`

Emitted when training fails or is cancelled.

```json
{
  "type": "train_failed",
  "job_id": "a1b2c3d4e5f6",
  "error": "No feature data found for the given date range"
}
```

## Auto-Reload

When training completes via the web UI while a tray instance is
running, the newly trained model is automatically loaded as the active
suggester. No manual model-switch is required.

## Frontend

The training UI is accessible via the "training" tab in the StatePanel.
It provides:

- **Data Readiness** — date range picker and data check showing feature
  coverage and label span counts.
- **Train Form** — configurable boost rounds, class weight, synthetic
  data toggle, and a "Train Model" button.
- **Progress Indicator** — real-time step, message, and progress bar
  fed by `train_progress` WebSocket events.
- **Result Display** — macro/weighted F1 on completion with the model
  directory path.
- **Model List** — all valid bundles sorted by creation date with
  metrics.
