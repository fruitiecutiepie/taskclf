import type { OverwritePending } from "../components/LabelOverwrite";
import { parseISODate } from "./date";

export interface TimeSelection {
  selectedMinutes: number;
  fillFromLast: boolean;
  lastLabelEndTs: string | null;
  extendFwd: boolean;
}

/**
 * Recalculate an overwrite-pending state after the user changes the time
 * picker. Returns the updated pending, or `null` if no conflicts remain.
 *
 * `now` is the current wall-clock time — callers should pass `new Date()`.
 */
export function labelOverwritePendingUpdGet(
  pending: OverwritePending,
  sel: TimeSelection,
  now: Date,
): OverwritePending | null {
  let start: Date;
  if (sel.fillFromLast && sel.lastLabelEndTs) {
    start = parseISODate(sel.lastLabelEndTs);
  } else if (sel.selectedMinutes === 0) {
    start = now;
  } else {
    start = new Date(now.getTime() - sel.selectedMinutes * 60_000);
  }

  const startMs = start.getTime();
  const endMs = now.getTime();
  const remaining = pending.conflicts.filter((c) => {
    const cs = parseISODate(c.start_ts).getTime();
    const ce = parseISODate(c.end_ts).getTime();
    return cs < endMs && startMs < ce;
  });

  if (remaining.length === 0) {
    return null;
  }

  return {
    ...pending,
    start: start.toISOString(),
    end: now.toISOString(),
    conflicts: remaining,
    extendForward: sel.selectedMinutes === 0 ? true : sel.extendFwd,
  };
}
