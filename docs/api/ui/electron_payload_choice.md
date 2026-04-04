# electron.payload_choice

Pure helpers used by the Electron main process to build compatible payload
version lists for the tray **Payload** submenu and for optional **Choose
Version** prompts during install/update flows.

## Functions

- `orderedCompatiblePayloadVersions(payloads, compatiblePayloads)` — versions
  from the payload index that satisfy the launcher compatibility range, newest
  first (same ordering as the tray menu).
- `installableOnlyPayloadVersions(compatibleOrderedDescending, installedVersions)`
  — compatible versions that are not yet installed locally.
- `payloadChooserOffersMultipleVersions(compatibleOrderedDescending)` — `true`
  when more than one compatible version exists (the optional chooser is only
  offered in that case).

## Integration

- Implemented in [`electron/payload_choice.ts`](../../../electron/payload_choice.ts).
- Uses [`electron/update_policy.ts`](../../../electron/update_policy.ts) for
  compatibility checks and version ordering.
- Consumed by [`electron/main.ts`](../../../electron/main.ts) together with
  [`electron/updater.ts`](../../../electron/updater.ts).
- Covered by `electron/payload_choice.test.js` (run `make electron-test`).
