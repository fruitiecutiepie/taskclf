import type { OverwritePending } from "../components/LabelOverwrite";
import { iso_date_parse } from "./date";

export type TimeSelection = {
  selectedMinutes: number;
  fillFromLast: boolean;
  lastLabelEndTs: string | null;
  extendFwd: boolean;
};

/**
 * Recalculate an overwrite-pending state after the user changes the time
 * picker. Returns the updated pending, or `null` if no conflicts remain.
 *
 * `now` is the current wall-clock time — callers should pass `new Date()`.
 */
export function label_overwrite_pending_upd_get(
  pending: OverwritePending,
  sel: TimeSelection,
  now: Date,
): OverwritePending | null {
  let start: Date;
  if (sel.fillFromLast && sel.lastLabelEndTs) {
    start = iso_date_parse(sel.lastLabelEndTs);
  } else if (sel.selectedMinutes === 0) {
    start = now;
  } else {
    start = new Date(now.getTime() - sel.selectedMinutes * 60_000);
  }

  const startMs = start.getTime();
  const endMs = now.getTime();
  const remaining = pending.conflicts.filter((c) => {
    const cs = iso_date_parse(c.start_ts).getTime();
    const ce = iso_date_parse(c.end_ts).getTime();
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
