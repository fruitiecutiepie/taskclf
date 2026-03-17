# ui.window

WindowAPI and WindowChild for pywebview-based floating UI.

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
Child windows (label grid, panel) are anchored below the pill on
initial show.  Once visible, children can be freely dragged to any
monitor; they will not snap back until hidden and re-shown.

## WindowChild

Encapsulates the visibility / pin / timer state machine shared by the
label-grid and state-panel child windows.  Each `WindowChild` instance
holds its own `window` reference, visibility and pin flags, a
delayed-hide timer, and an expected-position tuple for drag detection.

Positioning logic is injected via a callback (`position_fn`) so each
child can use different layout math while sharing the same state
machine.

### Methods

| Method | Description |
|--------|-------------|
| `visibility_on(main)` | Show on hover (non-pinned); cancels any pending hide timer |
| `visibility_off_deferred()` | Schedule a delayed hide (300 ms) unless pinned |
| `visibility_off()` | Immediate hide — clears visible, pinned, and expected position |
| `pin_toggle(main)` | Toggle pinned state (click to pin/unpin) |
| `timer_cancel()` | Cancel any pending hide timer |
| `position_sync()` | Reposition via the injected layout callback |
| `drag_detected()` | True if the user has dragged the window away from expected position |

## WindowAPI

Python API exposed to JavaScript as `window.pywebview.api.<method>()`.
The `Host` adapter in the frontend's `host.ts` calls these methods;
components never reference `window.pywebview` directly.

Internally, `WindowAPI` delegates to two `WindowChild` instances
(`_label` and `_panel`) for the label grid and state panel.

### Public methods

| Method | Description |
|--------|-------------|
| `bind(window)` | Bind the main compact window and subscribe to its `moved` event |
| `bind_label(label)` | Bind the label grid window |
| `bind_panel(panel)` | Bind the state panel window |
| `window_toggle()` | Toggle the compact pill's visibility |
| `label_grid_show()` | Show the label grid below the pill (right-aligned) |
| `label_grid_hide()` | Schedule a delayed hide of the label grid (300 ms) |
| `state_panel_toggle()` | Toggle the panel's visibility; positions below the label grid when both are visible |
| `visible` | Property returning whether the compact pill is currently visible |

### Window positioning

- The label grid is placed at `(pill.x + pill.width - grid.width, pill.y + pill.height + 4)`, right-aligned with the pill.
- The panel is placed below the pill (or below the label grid if it is visible), also right-aligned.
- When the pill is dragged, the label grid follows *only if the user has not independently dragged it*.  Once the user drags a child window away (beyond a 10 px tolerance), that child stops following and stays where the user placed it.
- Hiding a child window resets its expected position, so the next show re-anchors it to the pill.
- Child windows hide with a 300 ms delay to avoid flicker on rapid toggle.

### Dragging

All three windows use CSS `pywebview-drag-region` elements for drag
handles (the pill's entire surface and a small grab bar at the top of
each child window).  The main pill sets `easy_drag=False` to avoid
conflicts between the native easy-drag handler and the CSS drag region,
which previously caused glitches on multi-monitor setups.

## window_run (ui.window_run)

```python
window_run(
    port: int = 8741,
    on_ready: Callable[..., Any] | None = None,
    window_api: WindowAPI | None = None,
) -> None
```

Creates all three pywebview windows and starts the GUI loop.
**Blocks on the main thread** until the user closes the window.

Defined in `taskclf.ui.window_run`.

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

::: taskclf.ui.window_run
