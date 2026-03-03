# ui.window

Native floating window via pywebview.

## Overview

Creates frameless, always-on-top, draggable windows backed by the
platform webview (WebKit on macOS, Edge WebView2 on Windows).  Exposes
a `WindowAPI` to the SolidJS frontend via `window.pywebview.api`.

Three windows are managed:

| Window | Size (w x h) | Purpose |
|--------|--------------|---------|
| Compact pill | 150 x 30 | Persistent header badge showing current label/app |
| Label grid | 280 x 330 | Quick-label popup (hidden by default) |
| State panel | 280 x 520 | System/history debug panel (hidden by default) |

The compact pill is positioned at the top-right of the primary screen.
Child windows (label grid, panel) are anchored below the pill and
reposition automatically when the pill is dragged.

## WindowAPI

Python API exposed to JavaScript as `window.pywebview.api.<method>()`.
The `Host` adapter in the frontend's `host.ts` calls these methods;
components never reference `window.pywebview` directly.

### Public methods

| Method | Description |
|--------|-------------|
| `bind(window)` | Bind the main compact window and subscribe to its `moved` event |
| `bind_label(label)` | Bind the label grid window |
| `bind_panel(panel)` | Bind the state panel window |
| `toggle_window()` | Toggle the compact pill's visibility |
| `show_label_grid()` | Show the label grid below the pill (right-aligned) |
| `hide_label_grid()` | Schedule a delayed hide of the label grid (300 ms) |
| `toggle_state_panel()` | Toggle the panel's visibility; positions below the label grid when both are visible |
| `visible` | Property returning whether the compact pill is currently visible |

### Window positioning

- The label grid is placed at `(pill.x + pill.width - grid.width, pill.y + pill.height + 4)`, right-aligned with the pill.
- The panel is placed below the pill (or below the label grid if it is visible), also right-aligned.
- When the pill is dragged, the label grid automatically repositions via the `moved` event handler.
- Child windows hide with a 300 ms delay to avoid flicker on rapid toggle.

## run_window

```python
run_window(
    port: int = 8741,
    on_ready: Callable[..., Any] | None = None,
    window_api: WindowAPI | None = None,
) -> None
```

Creates all three pywebview windows and starts the GUI loop.
**Blocks on the main thread** until the user closes the window.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `port` | `8741` | Port of the FastAPI server to load in the webview |
| `on_ready` | `None` | Callback invoked after the GUI loop starts (receives the window object) |
| `window_api` | `None` | Shared `WindowAPI` instance; a new one is created when `None` |

The main window loads `http://127.0.0.1:{port}`, the label grid loads
`?view=label`, and the panel loads `?view=panel`.  All windows are
created with `frameless=True`, `on_top=True`, `transparent=True`.

On macOS, stderr is temporarily redirected to `/dev/null` during
pywebview startup to suppress noisy WebKit warnings, then restored
in the startup callback.

## Integration

- Used by the [`ui` CLI command](../cli/main.md) and
  [`ui.tray`](labeling.md) to launch the native window.
- The `WindowAPI` instance is shared with the FastAPI app so that
  REST endpoints (e.g. `POST /api/window/show-label-grid`) can
  control window visibility.
- Events flow from `ActivityMonitor` through
  [`EventBus`](events.md) to the WebSocket layer inside the
  webview.

::: taskclf.ui.window
