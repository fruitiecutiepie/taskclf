# adapters/

Integrations with external tools or OS APIs.

## Subpackages

### `activitywatch/`
- `types.py` -- `AWEvent` (window events) and `AWInputEvent` (keyboard/mouse
  aggregate counts from `aw-watcher-input`).
- `mapping.py` -- App-name normalization, browser/editor/terminal classification,
  and semantic `app_category` assignment (browser, editor, terminal, chat, email,
  meeting, docs, design, devtools, media, file_manager, utilities, project_mgmt,
  other).
- `client.py` -- AW JSON export parser and REST API client.  Supports both
  `currentwindow` (window watcher) and `os.hid.input` (input watcher) bucket
  types for file-based and REST-based ingestion.

### `input/`
- Optional OS-specific input aggregators (counts only, not yet implemented).
  The `aw-watcher-input` integration in `activitywatch/` covers the same
  signals when ActivityWatch is running.

## Invariants
- Adapters must output normalized events satisfying the `core.types.Event` protocol.
- Raw window titles are never persisted -- they are replaced with salted hashes.
- App names are mapped to reverse-domain identifiers via the known-app registry.
- Input events carry only aggregate counts (presses, clicks, movement, scroll)
  -- never individual key identities.
- Keep adapter-specific quirks out of `core/`.
- Adapters should be swappable without changing feature or model code.

## Event Protocol
The `Event` protocol (`core.types.Event`) defines the minimal attribute set
that any adapter event must expose.  `AWEvent` satisfies this protocol
structurally (no inheritance required).  New adapters should likewise expose
`timestamp`, `duration_seconds`, `app_id`, `window_title_hash`, `is_browser`,
`is_editor`, `is_terminal`, and `app_category`.

`AWInputEvent` is a separate type that does not implement the `Event` protocol
-- it feeds into the feature builder as a supplementary data source via the
`input_events` parameter of `build_features_from_aw_events()`.
