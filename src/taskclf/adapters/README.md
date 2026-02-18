# adapters/

Unstable integrations with external tools or OS APIs.

## Examples
- `activitywatch/`: reading ActivityWatch exports or API
- `input/`: optional OS-specific input aggregators (counts only)

## Invariants
- Adapters must output normalized `core.types.Event` streams (or equivalent).
- Keep adapter-specific quirks out of `core/`.
- Adapters should be swappable without changing feature or model code.
