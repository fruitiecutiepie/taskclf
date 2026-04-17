# electron.node_http

Internal Node-backed HTTP helper for Electron main-process requests.

## Overview

`electron/node_http.ts` provides a narrow `nodeFetch(...)` helper that uses
Node's built-in `http` / `https` transports instead of Electron's
`net.fetch()`.

The wrapper exists so launcher update checks can keep fetch-like call sites
(`Response`, `.ok`, `.json()`, streaming bodies) without routing through
Electron's Chromium networking layer.

## Behavior

- accepts standard `RequestInit` fields plus `maxRedirects`
- follows HTTP redirects (`301`, `302`, `303`, `307`, `308`) by default
- preserves ordinary non-2xx responses instead of throwing
- honors `AbortSignal` cancellation
- normalizes response headers into a standard `Headers` object
- returns a standard `Response`, so callers can keep using `.json()`,
  `.text()`, `.arrayBuffer()`, and streaming `body`

The helper intentionally stays small and transport-focused. It does not own
update policy, payload validation, or tray business logic.

## Integration

- Used by `electron/updater.ts` for launcher manifest, payload index,
  payload manifest, and payload zip downloads
- Covered by `electron/node_http.test.js`
- Referenced by [`electron_shell`](electron_shell.md)
