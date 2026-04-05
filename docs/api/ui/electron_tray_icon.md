# electron.tray_icon

Tray-icon asset selection helpers for the Electron desktop shell.

## Overview

`electron/tray_icon.ts` keeps tray icon path selection separate from the main
Electron process so packaging behavior is explicit and testable.

The module distinguishes between:

- `icon.png` for the app and window icon on every platform
- `trayTemplate.png` and `trayTemplate@2x.png` for the macOS menu-bar tray icon

That separation matters because macOS status-bar items expect a monochrome
template image instead of the full-color app icon.

## Exports

### `getBuildAssetPath(fileName, baseDir=__dirname)`

Resolves an Electron build asset under `../build/` relative to a compiled
module directory.

### `getAppIconPath(baseDir=__dirname)`

Resolves the packaged app/window icon path (`build/icon.png`).

### `getTrayAssetFiles(platform=process.platform)`

Returns the relative build assets that must be present for the given platform.
On macOS this includes both `trayTemplate.png` and `trayTemplate@2x.png`.

### `getTrayIconAsset(baseDir=__dirname, platform=process.platform)`

Returns the tray icon path plus whether Electron should treat it as a template
image. `electron/main.ts` uses this to call `nativeImage.setTemplateImage(true)`
on macOS before constructing the tray.

## Related Files

- Implemented in [`electron/tray_icon.ts`](../../../electron/tray_icon.ts)
- Consumed by [`electron/main.ts`](../../../electron/main.ts)
- Asset generation lives in [`electron/scripts/build-icon.js`](../../../electron/scripts/build-icon.js)
- Launcher behavior is documented in [`electron_shell.md`](electron_shell.md)
