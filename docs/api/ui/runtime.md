# ui.runtime

Shared non-GUI runtime helpers used by both `taskclf ui` and `taskclf tray`.

- `ActivityMonitor` owns ActivityWatch polling, transition detection, and status
  event publishing.
- `_LabelSuggester` wraps online inference for interval label suggestions.
- Keeping these classes outside `taskclf.ui.tray` lets browser-only UI startup
  avoid importing tray icon dependencies such as Pillow and pystray.

::: taskclf.ui.runtime
