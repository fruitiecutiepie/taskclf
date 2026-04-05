# ui.error_banner

Dismissible frontend error callout used across the SolidJS UI.

Source:
[`src/taskclf/ui/frontend/src/components/ErrorBanner.tsx`](../../../src/taskclf/ui/frontend/src/components/ErrorBanner.tsx)

## Behavior

- Renders a compact inline error banner with the message body.
- Errors stay visible until the user clicks **Close**; they do not auto-dismiss.
- Includes **Copy error** to copy the raw error text to the clipboard.
- Uses `navigator.clipboard.writeText(...)` when available and falls back to a
  temporary hidden textarea plus `document.execCommand("copy")` for embedded
  runtimes that do not expose the async Clipboard API.

## Props

| Prop | Type | Description |
|---|---|---|
| `message` | `string` | Error text shown to the user and copied verbatim by the copy action |
| `on_close` | `() => void` | Optional close handler; when omitted, the banner is read-only |
