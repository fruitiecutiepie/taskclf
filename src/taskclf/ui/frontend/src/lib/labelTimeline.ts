import { parseDate } from "./date";

export interface LabelEntry {
  label: string;
  start_ts: string;
  end_ts: string;
}

export interface TimelineSegment {
  label: string | null;
  startMs: number;
  endMs: number;
  fraction: number;
}

export interface GapItem {
  kind: "gap";
  start_ts: string;
  end_ts: string;
}

export interface LabelItem {
  kind: "label";
  label: string;
  start_ts: string;
  end_ts: string;
}

export type TimelineItem = GapItem | LabelItem;

export function buildDayTimeline(
  entries: LabelEntry[],
  dayDateStr: string,
): { segments: TimelineSegment[]; items: TimelineItem[]; spanMs: number } {
  const dayStart = new Date(`${dayDateStr}T00:00:00`).getTime();
  const dayEnd = new Date(`${dayDateStr}T23:59:59.999`).getTime();
  const spanMs = dayEnd - dayStart;

  if (!entries.length) {
    const seg: TimelineSegment = { label: null, startMs: dayStart, endMs: dayEnd, fraction: 1 };
    const gap: GapItem = { kind: "gap", start_ts: new Date(dayStart).toISOString(), end_ts: new Date(dayEnd).toISOString() };
    return { segments: [seg], items: [gap], spanMs };
  }

  const sorted = [...entries].sort((a, b) => parseDate(a.start_ts).getTime() - parseDate(b.start_ts).getTime());
  const segments: TimelineSegment[] = [];
  const items: TimelineItem[] = [];
  let cursor = dayStart;

  for (const entry of sorted) {
    const s = Math.max(parseDate(entry.start_ts).getTime(), dayStart);
    const e = Math.min(parseDate(entry.end_ts).getTime(), dayEnd);
    if (e <= cursor) continue;

    if (s > cursor) {
      segments.push({ label: null, startMs: cursor, endMs: s, fraction: (s - cursor) / spanMs });
      items.push({ kind: "gap", start_ts: new Date(cursor).toISOString(), end_ts: new Date(s).toISOString() });
    }
    const segStart = Math.max(s, cursor);
    segments.push({ label: entry.label, startMs: segStart, endMs: e, fraction: (e - segStart) / spanMs });
    items.push({ kind: "label", label: entry.label, start_ts: entry.start_ts, end_ts: entry.end_ts });
    cursor = e;
  }

  if (cursor < dayEnd) {
    segments.push({ label: null, startMs: cursor, endMs: dayEnd, fraction: (dayEnd - cursor) / spanMs });
    items.push({ kind: "gap", start_ts: new Date(cursor).toISOString(), end_ts: new Date(dayEnd).toISOString() });
  }

  return { segments, items, spanMs };
}

export function itemKey(item: TimelineItem): string {
  return `${item.kind}|${item.start_ts}|${item.end_ts}`;
}
