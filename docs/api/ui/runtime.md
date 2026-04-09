# ui.runtime

Shared non-GUI runtime helpers used by both `taskclf ui` and `taskclf tray`.

- `ActivityMonitor` owns activity-source polling, transition detection, cached
  provider diagnostics, and status event publishing.
- `_LabelSuggester` wraps online inference for interval label suggestions.
- Keeping these classes outside `taskclf.ui.tray` lets browser-only UI startup
  avoid importing tray icon dependencies such as Pillow and pystray.

## Activity source status

`ActivityMonitor` now publishes a provider-neutral `activity_provider` object in
every WebSocket `status` event. The object is initialized in `checking` state
immediately at startup, then updated to:

- `ready` after a successful initial probe
- `setup_required` after the first failed startup probe, so the UI can show
  setup guidance without blocking manual labeling

Legacy `aw_*` fields remain in the status payload temporarily for compatibility,
but new UI code should prefer `status.activity_provider`.

::: taskclf.ui.runtime
