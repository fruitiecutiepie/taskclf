# ui.host_window_drag_strip

SolidJS component `HostWindowDragStrip` in
[`src/taskclf/ui/frontend/src/components/HostWindowDragStrip.tsx`](../../../src/taskclf/ui/frontend/src/components/HostWindowDragStrip.tsx).

## Role

Renders the small top grab bar on the label-grid and state-panel routes
(`?view=label`, `?view=panel`). Styling depends on the active host:

| Host | Behavior |
|------|----------|
| `pywebview` | Applies CSS class `pywebview-drag-region` for webview drag. |
| `electron` | Sets `-webkit-app-region` / `app-region` to `drag` for Electron window dragging. |
| `browser` | Visual bar only; no native drag integration. |

## Related

- Native window dragging overview: [`window.md`](window.md) (Dragging section).
- Host command mapping: [`src/taskclf/ui/frontend/src/lib/host.ts`](../../../src/taskclf/ui/frontend/src/lib/host.ts).
