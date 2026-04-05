# electron.tray_dashboard

Pure helpers for the Electron shell's tray-to-dashboard visibility policy.

## Functions

- `dashboardWindowActionForTrayInteraction(interaction)` — returns
  `"show"` for a primary tray icon click and `"toggle"` for the explicit
  tray-menu **Toggle Dashboard** action.

## Integration

- Implemented in [`electron/tray_dashboard.ts`](../../../electron/tray_dashboard.ts).
- Consumed by [`electron/main.ts`](../../../electron/main.ts).
- Covered by `electron/tray_dashboard.test.js` (run `make electron-test`).
