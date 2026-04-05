# Frontend date helpers

Source: [`src/taskclf/ui/frontend/src/lib/date.ts`](../../../src/taskclf/ui/frontend/src/lib/date.ts)

## `gap_shortcut_label_from_end(end_ms: number, now_ms: number): string | null`

Builds the quick-label **gap** button label (e.g. `gap 5m`, `gap 1h30m`) from a label span’s end time in milliseconds and the current wall time in milliseconds. Returns `null` when the gap is under one rounded minute (same visibility rule as before: hidden until at least ~1 minute of unlabeled time).

Used by [`LabelTimePicker`](../../../src/taskclf/ui/frontend/src/components/LabelTimePicker.tsx) together with a periodic wall-clock tick so the text stays current while the picker remains open.

## See also

- [`time_ago`](../../../src/taskclf/ui/frontend/src/lib/date.ts) — relative “Nm ago” strings for the last-label footer (still sampled at render; not driven by the gap tick).
