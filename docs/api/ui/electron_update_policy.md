# electron.update_policy

Internal Electron launcher policy helpers for payload selection.

## Overview

`electron/update_policy.ts` resolves the newest payload compatible with the
current launcher release and decides whether the app can keep using the active
payload, switch to an installed compatible version, or download one.

It provides small pure helpers for:

- `manifestUrlForLauncherVersion(version, overrideUrl?)` — returns the default
  GitHub manifest URL for `launcher-v<version>`, unless
  `TASKCLF_MANIFEST_URL` supplies an explicit override.
- `compareVersions(...)` — compares `X.Y.Z` payload versions numerically.
- `isPayloadVersionCompatible(...)` — checks a payload version against the
  launcher manifest's compatibility range.
- `selectLatestCompatiblePayloadVersion(...)` — picks the newest payload from
  the payload index that satisfies the launcher's compatibility range.
- `resolvePayloadSyncPlan(...)` — classifies launcher/payload state as:
  `none`, `switch`, or `download`.

The module exists so the updater can make the release-channel decision in one
place and so the policy can be unit-tested without pulling in Electron runtime
state.

## Behavior

- If the active payload already matches the desired payload version and its
  backend executable exists, the policy returns `none`.
- If the desired payload is already cached locally but `active.json` points at
  another version, the policy returns `switch`.
- If the desired payload is not installed, the policy returns `download`.

## Integration

- Used by [`electron_shell`](electron_shell.md) via `electron/updater.ts`
- Covered by `electron/update_policy.test.js`
