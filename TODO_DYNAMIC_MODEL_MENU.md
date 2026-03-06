# TODO — Dynamic Model Submenu Refresh

## Problem

The Model submenu is built once at tray startup via `_build_model_submenu()`.
When a retrain creates a new model bundle in `models/`, the submenu is stale
until the user restarts the tray. "Reload Model" does not help because it
re-reads the *currently loaded* bundle, not the directory listing.

## Desired Behavior

After a retrain (or any change to `models_dir`), the Model submenu should
reflect the current set of valid bundles without requiring a restart.

## Approach Options

### Option A: Rebuild menu on every open (recommended)

pystray supports dynamic menus by passing a callable to the `menu` parameter
of `pystray.Icon`. Instead of building the menu once:

```python
# Current (static)
self._icon = pystray.Icon("taskclf", icon_image, "taskclf", menu=self._build_menu())

# Proposed (dynamic)
self._icon = pystray.Icon("taskclf", icon_image, "taskclf", menu=self._build_menu)
```

When `menu` is a callable, pystray calls it every time the menu is about to be
shown. This means `_build_model_submenu()` re-scans `models_dir` on each
right-click — picking up new bundles automatically.

**Pros:** Simple, no filesystem watchers, no polling, no state to manage.
**Cons:** Slight delay on right-click if `models_dir` has many entries (should
be negligible — `list_bundles` just reads `metadata.json` per subdirectory).

### Option B: Filesystem watcher + menu rebuild

Use `watchdog` or `inotify` to watch `models_dir` for new subdirectories and
trigger a menu rebuild. More complex, adds a dependency, and pystray's menu
update mechanism varies by platform.

### Option C: "Refresh Models" menu item

Add a manual "Refresh Models" item to the submenu that re-scans and rebuilds.
Less automatic but simple.

## Recommendation

**Option A** is the simplest and most robust. The key change is passing
`self._build_menu` (the method reference) instead of `self._build_menu()`
(the result) to `pystray.Icon`.

## Changes Required

- [ ] In `TrayLabeler.run()`, change `menu=self._build_menu()` to
      `menu=self._build_menu` so pystray rebuilds the menu on each open.
- [ ] Verify this works on macOS (AppKit backend) and Linux (AppIndicator/GTK).
      Some backends may not support callable menus — add a fallback if needed.
- [ ] Update tests: `_build_menu` is now called repeatedly, ensure no side
      effects or stale state accumulates.
- [ ] Update docs to mention that the Model submenu auto-refreshes.

## Impact

Once implemented, "Reload Model" becomes useful only for the rare case where
the *contents* of the currently loaded bundle directory changed on disk (e.g.
manual file replacement). The submenu itself handles new/removed bundles
automatically.
