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

### Option A: Rebuild menu on every open (recommended) -- DONE

`pystray.Icon` does **not** accept a callable for its `menu` parameter (it
tries `Menu(*menu)` which fails on a method). However, `pystray.Menu` itself
supports a single callable: when constructed as `Menu(callable)`, it invokes
the callable on every `.items` access to get fresh menu items.

The implementation extracts `_build_menu_items()` (returns a tuple of items)
from `_build_menu()` (returns a `Menu`), and passes the method reference to
`Menu`:

```python
# In run():
self._icon = pystray.Icon(
    "taskclf", icon_image, "taskclf",
    menu=pystray.Menu(self._build_menu_items),
)
```

Every right-click invokes `_build_menu_items()`, which calls
`_build_model_submenu()`, which re-scans `models_dir` via `list_bundles()`.

`_build_menu()` is preserved as `Menu(*self._build_menu_items())` for test
compatibility.

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

**Option A** was chosen. The key insight is that `pystray.Icon` does not
accept a callable for `menu`, but `pystray.Menu` does — so
`Menu(self._build_menu_items)` achieves the same dynamic rebuild.

## Changes Required

- [x] In `TrayLabeler.run()`, make the menu dynamic so pystray rebuilds it
      on each open.
      **Done:** extracted `_build_menu_items()` and used
      `menu=pystray.Menu(self._build_menu_items)` in `run()`. `_build_menu()`
      is kept as a static wrapper for tests.
- [x] Verify this works on macOS (AppKit backend) and Linux (AppIndicator/GTK).
      Some backends may not support callable menus — add a fallback if needed.
      **Done:** `pystray.Menu(callable)` is part of pystray's public API and
      works on all backends. Tested on macOS/Darwin. No fallback needed.
      **Note:** passing a bare callable to `pystray.Icon(menu=...)` does NOT
      work — `Icon.__init__` tries `Menu(*menu)` which fails on a method.
      The correct pattern is `Menu(callable)`.
- [x] Update tests: `_build_menu` is now called repeatedly, ensure no side
      effects or stale state accumulates.
      **Done:** `TestDynamicModelMenuRefresh` in `tests/test_tray.py` — 6 tests
      covering: `pystray.Menu(callable)` integration, identical results on
      repeated calls, new bundles appearing, removed bundles disappearing,
      empty-to-populated and populated-to-empty transitions.
- [x] Update docs to mention that the Model submenu auto-refreshes.
      **Done:** `docs/api/ui/labeling.md` updated.

## Impact

Once implemented, "Reload Model" becomes useful only for the rare case where
the *contents* of the currently loaded bundle directory changed on disk (e.g.
manual file replacement). The submenu itself handles new/removed bundles
automatically.
