# infer.baseline

Rule-based baseline classifier (no ML).

Applies heuristic rules to feature windows in priority order:

0. **BreakIdle** — lockscreen / OS login window (`app_category == "lockscreen"`)
1. **BreakIdle** — near-zero activity or long idle run
2. **ReadResearch** — browser foreground with high scroll and low typing
3. **Build** — editor/terminal foreground with high typing and shortcuts
4. **Mixed/Unknown** — fallback reject label

Rule 0 fires when the foreground app is an OS lock screen (macOS
`loginwindow`, Windows `LockApp.exe`, Linux screen lockers).  Since no
productive task is possible while the screen is locked, this overrides
all other signals unconditionally.

This establishes the cold-start performance floor that ML models must beat.

## Thresholds

All thresholds are defined in [`taskclf.core.defaults`](../core/defaults.md) and can be overridden
per-call:

| Constant | Default | Used by |
|----------|---------|---------|
| `BASELINE_IDLE_ACTIVE_THRESHOLD` | 5.0 s | BreakIdle rule |
| `BASELINE_IDLE_RUN_THRESHOLD` | 50.0 s | BreakIdle rule |
| `BASELINE_SCROLL_HIGH` | 3.0 events/min | ReadResearch rule |
| `BASELINE_KEYS_LOW` | 10.0 keys/min | ReadResearch rule |
| `BASELINE_KEYS_HIGH` | 30.0 keys/min | Build rule |
| `BASELINE_SHORTCUT_HIGH` | 1.0 shortcuts/min | Build rule |

::: taskclf.infer.baseline
