# taskclf

![taskclf icon](../assets/taskclf-icon.svg){ width="160" }

A local-first, privacy-preserving classifier that infers your task type
(coding, writing, meetings, etc.) from computer activity signals.

!!! info "Privacy first"
    No raw keystrokes or window titles are stored — only aggregate
    rates and hashed identifiers. Everything runs on your machine.

**Who is this for?** Developers and power users who want automated,
private time tracking without sending data to a third-party service.

## Quick start

```bash
uv sync
uv run taskclf --help
```

See the [Getting Started](guide/usage.md) guide for full setup,
or try the [synthetic data demo](guide/usage.md#use-case-2-quick-demo-with-synthetic-data)
to run the full pipeline without ActivityWatch.

## Key concepts

| Concept | Summary |
|---|---|
| **Buckets** | Activity aggregated into 60 s windows — [Time & Buckets](guide/time_spec.md) |
| **8 core labels** | Build, Debug, Review, Write, ReadResearch, Communicate, Meet, BreakIdle — [Task Labels](guide/labels_v1.md) |
| **Privacy by design** | No raw content is ever stored — [Privacy Model](guide/privacy.md) |
| **Custom taxonomy** | Map core labels to your own categories without retraining — [Custom Taxonomy](guide/taxonomy.md) |

## Guides

- [Getting Started](guide/usage.md) — workflows from ingestion to daily reports
- [Privacy Model](guide/privacy.md) — what is and isn't stored
- [Inference Pipeline](guide/model_io.md) — how a feature row becomes a prediction
- [Acceptance Criteria](guide/acceptance.md) — quality gates for model promotion

## API Reference

Auto-generated from source docstrings. Use the **API Reference** tab for the
full listing, or jump to a group:

| Group | Description |
|---|---|
| [Core](api/core/types.md) | Types, schemas, storage, validation, and observability |
| [Adapters](api/adapters/activitywatch.md) | ActivityWatch data import |
| [Features](api/features/build.md) | Feature extraction from raw events |
| [Labels](api/labels/store.md) | Label storage, projection, and active labeling |
| [Training](api/train/build_dataset.md) | Dataset construction, model training, and evaluation |
| [Inference](api/infer/batch.md) | Prediction, calibration, smoothing, and monitoring |
| [Reports](api/report/daily.md) | Daily summaries and export |
| [UI](api/ui/labeling.md) | Web dashboard and system tray app |
| [CLI](api/cli/main.md) | Command-line interface |
