# adapters.activitywatch

ActivityWatch adapter: data ingestion from AW JSON exports and the AW REST API.

## Error handling

All REST API functions raise typed exceptions so callers can distinguish
failure modes and respond accordingly:

- **`AWConnectionError`** -- the AW server is unreachable or refused the
  connection (wraps `ConnectionRefusedError`, `ConnectionResetError`,
  `urllib.error.URLError`).
- **`AWTimeoutError`** -- the AW server did not respond within the
  configured timeout (wraps `TimeoutError`, `socket.timeout`).

All REST functions accept an optional `timeout` keyword argument
(default: `DEFAULT_AW_TIMEOUT_SECONDS = 10`).  The tray app exposes
this as the `aw_timeout_seconds` config setting and `--aw-timeout` CLI
flag.

When the tray polls AW and encounters repeated failures, it applies
**adaptive backoff**: connection-refused errors progressively increase
the sleep between polls (exponential, capped at 5 minutes), while
timeout errors keep the normal polling interval (since the timeout
itself is already the wait).  After 3 consecutive failures a WARNING
is logged and an `aw_unreachable` status event is published to the
web UI.  On recovery, an INFO message is logged and the backoff resets.

## types

::: taskclf.adapters.activitywatch.types

## mapping

::: taskclf.adapters.activitywatch.mapping

## client

::: taskclf.adapters.activitywatch.client
