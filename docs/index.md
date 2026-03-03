# taskclf

A local-first, privacy-preserving classifier that infers your task type
(coding, writing, meetings, etc.) from computer activity signals.

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

- **Buckets** — activity is aggregated into 60-second windows ([Time & Buckets](guide/time_spec.md))
- **8 core labels** — Build, Debug, Review, Write, ReadResearch, Communicate, Meet, BreakIdle ([Task Labels](guide/labels_v1.md))
- **Privacy by design** — no raw content is ever stored ([Privacy Model](guide/privacy.md))
- **Custom taxonomy** — map core labels to your own categories without retraining ([Custom Taxonomy](guide/taxonomy.md))

## Guides

- [Getting Started](guide/usage.md) — workflows from ingestion to daily reports
- [Privacy Model](guide/privacy.md) — what is and isn't stored
- [Inference Pipeline Overview](guide/model_io.md) — how a feature row becomes a prediction
- [Acceptance Criteria](guide/acceptance.md) — quality gates for model promotion

## API Reference

Auto-generated from source docstrings. See the sidebar for the full
listing, or jump to a group:

- [Core](api/core/defaults.md) — shared types, schemas, storage, and validation
- [Adapters](api/adapters/activitywatch.md) — ActivityWatch data import
- [Features](api/features/build.md) — feature extraction from raw events
- [Labels](api/labels/store.md) — label storage, projection, and active labeling
- [Training](api/train/build_dataset.md) — dataset construction, model training, and evaluation
- [Inference](api/infer/batch.md) — prediction, calibration, smoothing, and monitoring
- [Reports](api/report/daily.md) — daily summaries and export
- [UI](api/ui/labeling.md) — web dashboard and system tray app
- [CLI](api/cli/main.md) — command-line interface
