# electron.update_policy

Internal Electron launcher policy helpers for payload selection.

## Overview

`electron/update_policy.ts` keeps the packaged launcher on the payload built
for the same launcher release.

It provides two small pure helpers:

- `manifestUrlForLauncherVersion(version, overrideUrl?)` — returns the default
  GitHub manifest URL for `launcher-v<version>`, unless
  `TASKCLF_MANIFEST_URL` supplies an explicit override.
- `resolvePayloadSyncPlan(...)` — classifies launcher/payload state as:
  `none`, `switch`, or `download`.

The module exists so the updater can make the release-channel decision in one
place and so the policy can be unit-tested without pulling in Electron runtime
state.

## Behavior

- If the active payload already matches the launcher version and its backend
  executable exists, the policy returns `none`.
- If the launcher-matched payload is already cached locally but `active.json`
  points elsewhere, the policy returns `switch`.
- If the launcher-matched payload is not installed, the policy returns
  `download`.

## Integration

- Used by [`electron_shell`](electron_shell.md) via `electron/updater.ts`
- Covered by `electron/update_policy.test.js`
