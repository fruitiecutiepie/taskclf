# ui.activity_provider

Provider-neutral activity-source helpers for UI-facing runtime and summary
surfaces.

- `ActivityWatcherProvider` defines the small contract the UI needs from an
  activity source: availability probing, source discovery, event fetches, and
  recent app summaries.
- `ActivityWatchProvider` is the current bundled implementation. Provider
  selection is still internal; the existing `aw_host` config remains the only
  user-facing knob.
- `ActivityProviderStatus` is the shared payload used by the runtime monitor,
  WebSocket `status.activity_provider`, and `GET /api/activity/summary`.
- Setup guidance strings are generated server-side so the frontend can render a
  consistent callout without hard-coding ActivityWatch copy.

The provider status object uses:

- `state`: `checking | ready | setup_required`
- `summary_available`: whether live summaries are currently available
- `endpoint` and `source_id`: diagnostics for the configured source
- `last_sample_count` and `last_sample_breakdown`: last successful sample
  diagnostics
- `setup_title`, `setup_message`, `setup_steps`, `help_url`: user-facing setup
  guidance for unavailable providers

::: taskclf.ui.activity_provider
