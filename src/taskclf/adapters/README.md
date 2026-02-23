# adapters/

Integrations with external tools or OS APIs.

## Subpackages

### `activitywatch/`
- `types.py` -- `AWEvent` Pydantic model (privacy-safe normalized event)
- `mapping.py` -- App-name normalization and browser/editor/terminal classification
- `client.py` -- AW JSON export parser and REST API client

### `input/`
- Optional OS-specific input aggregators (counts only, not yet implemented)

## Invariants
- Adapters must output normalized events satisfying the `core.types.Event` protocol.
- Raw window titles are never persisted -- they are replaced with salted hashes.
- App names are mapped to reverse-domain identifiers via the known-app registry.
- Keep adapter-specific quirks out of `core/`.
- Adapters should be swappable without changing feature or model code.

## Event Protocol
The `Event` protocol (`core.types.Event`) defines the minimal attribute set
that any adapter event must expose.  `AWEvent` satisfies this protocol
structurally (no inheritance required).  New adapters should likewise expose
`timestamp`, `duration_seconds`, `app_id`, `window_title_hash`, `is_browser`,
`is_editor`, and `is_terminal`.
