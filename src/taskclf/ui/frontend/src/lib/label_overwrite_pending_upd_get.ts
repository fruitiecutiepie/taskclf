import type { OverwritePending } from "../components/LabelOverwrite";
import { iso_date_parse } from "./date";

export type TimeSelection = {
  selected_minutes: number;
  fill_from_last: boolean;
  last_label_end_ts: string | null;
  extend_fwd: boolean;
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
  if (sel.fill_from_last && sel.last_label_end_ts) {
    start = iso_date_parse(sel.last_label_end_ts);
  } else if (sel.selected_minutes === 0) {
    start = now;
  } else {
    start = new Date(now.getTime() - sel.selected_minutes * 60_000);
  }

  const start_ms = start.getTime();
  const end_ms = now.getTime();
  const remaining = pending.conflicts.filter((c) => {
    const cs = iso_date_parse(c.start_ts).getTime();
    const ce = iso_date_parse(c.end_ts).getTime();
    return cs < end_ms && start_ms < ce;
  });

  if (remaining.length === 0) {
    return null;
  }

  return {
    ...pending,
    start: start.toISOString(),
    end: now.toISOString(),
    conflicts: remaining,
    extend_forward: sel.selected_minutes === 0 ? true : sel.extend_fwd,
  };
}
