import { date_parse } from "./date";

export type LabelEntry = {
  label: string;
  start_ts: string;
  end_ts: string;
};

export type TimelineSegment = {
  label: string | null;
  start_ms: number;
  end_ms: number;
  fraction: number;
};

export type GapItem = {
  kind: "gap";
  start_ts: string;
  end_ts: string;
};

export type LabelItem = {
  kind: "label";
  label: string;
  start_ts: string;
  end_ts: string;
};

export type TimelineItem = GapItem | LabelItem;

export function day_timeline_build(
  entries: LabelEntry[],
  day_date_str: string,
): { segments: TimelineSegment[]; items: TimelineItem[]; span_ms: number } {
  const day_start = new Date(`${day_date_str}T00:00:00`).getTime();
  const day_end = new Date(`${day_date_str}T23:59:59.999`).getTime();
  const span_ms = day_end - day_start;

  if (!entries.length) {
    const seg: TimelineSegment = {
      label: null,
      start_ms: day_start,
      end_ms: day_end,
      fraction: 1,
    };
    const gap: GapItem = {
      kind: "gap",
      start_ts: new Date(day_start).toISOString(),
      end_ts: new Date(day_end).toISOString(),
    };
    return { segments: [seg], items: [gap], span_ms };
  }

  const sorted = [...entries].sort(
    (a, b) => date_parse(a.start_ts).getTime() - date_parse(b.start_ts).getTime(),
  );
  const segments: TimelineSegment[] = [];
  const items: TimelineItem[] = [];
  let cursor = day_start;

  for (const entry of sorted) {
    const s = Math.max(date_parse(entry.start_ts).getTime(), day_start);
    const e = Math.min(date_parse(entry.end_ts).getTime(), day_end);
    if (e <= s) {
      continue;
    }

    if (s > cursor) {
      segments.push({
        label: null,
        start_ms: cursor,
        end_ms: s,
        fraction: (s - cursor) / span_ms,
      });
      items.push({
        kind: "gap",
        start_ts: new Date(cursor).toISOString(),
        end_ts: new Date(s).toISOString(),
      });
    }

    const seg_start = Math.max(s, cursor);
    if (seg_start < e) {
      segments.push({
        label: entry.label,
        start_ms: seg_start,
        end_ms: e,
        fraction: (e - seg_start) / span_ms,
      });
    } else {
      segments.push({ label: entry.label, start_ms: s, end_ms: e, fraction: 0 });
    }
    items.push({
      kind: "label",
      label: entry.label,
      start_ts: entry.start_ts,
      end_ts: entry.end_ts,
    });
    cursor = Math.max(cursor, e);
  }

  if (cursor < day_end) {
    segments.push({
      label: null,
      start_ms: cursor,
      end_ms: day_end,
      fraction: (day_end - cursor) / span_ms,
    });
    items.push({
      kind: "gap",
      start_ts: new Date(cursor).toISOString(),
      end_ts: new Date(day_end).toISOString(),
    });
  }

  return { segments, items, span_ms };
}

export function item_key(item: TimelineItem): string {
  const label = item.kind === "label" ? (item as LabelItem).label : "";
  return `${item.kind}|${item.start_ts}|${item.end_ts}|${label}`;
}
