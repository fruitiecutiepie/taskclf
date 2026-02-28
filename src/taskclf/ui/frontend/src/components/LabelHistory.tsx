import { type Accessor, type Component, createResource, createMemo, createSignal, For, Show } from "solid-js";
import { fetchLabels } from "../lib/api";
import { LABEL_COLORS } from "./StatePanel";

function parseDate(iso: string): Date {
  return new Date(iso.includes("Z") || iso.includes("+") ? iso : iso + "Z");
}

function isToday(d: Date): boolean {
  const now = new Date();
  return d.getFullYear() === now.getFullYear() && d.getMonth() === now.getMonth() && d.getDate() === now.getDate();
}

function fmtDate(d: Date): string {
  if (isToday(d)) return "Today";
  return d.toLocaleDateString([], { month: "short", day: "numeric" });
}

function fmtTime(d: Date): string {
  return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function fmtDuration(ms: number): string {
  const totalMin = Math.round(ms / 60_000);
  if (totalMin < 1) return "<1m";
  const h = Math.floor(totalMin / 60);
  const m = totalMin % 60;
  if (h === 0) return `${m}m`;
  if (m === 0) return `${h}h`;
  return `${h}h ${m}m`;
}

function dateKey(iso: string): string {
  const d = parseDate(iso);
  return `${d.getFullYear()}-${d.getMonth()}-${d.getDate()}`;
}

interface LabelEntry {
  label: string;
  start_ts: string;
  end_ts: string;
}

interface TimelineSegment {
  label: string | null;
  startMs: number;
  endMs: number;
  fraction: number;
}

interface DateGroup {
  dateLabel: string;
  entries: LabelEntry[];
  timeline: TimelineSegment[];
  spanMs: number;
}

function buildTimeline(entries: LabelEntry[]): { segments: TimelineSegment[]; spanMs: number } {
  if (!entries.length) return { segments: [], spanMs: 0 };

  const sorted = [...entries].sort((a, b) => parseDate(a.start_ts).getTime() - parseDate(b.start_ts).getTime());
  const rangeStart = parseDate(sorted[0].start_ts).getTime();
  const rangeEnd = Math.max(...sorted.map((e) => parseDate(e.end_ts).getTime()));
  const spanMs = rangeEnd - rangeStart;
  if (spanMs <= 0) return { segments: [], spanMs: 0 };

  const segments: TimelineSegment[] = [];
  let cursor = rangeStart;

  for (const entry of sorted) {
    const s = parseDate(entry.start_ts).getTime();
    const e = parseDate(entry.end_ts).getTime();

    if (s > cursor) {
      segments.push({ label: null, startMs: cursor, endMs: s, fraction: (s - cursor) / spanMs });
    }
    segments.push({ label: entry.label, startMs: Math.max(s, cursor), endMs: e, fraction: (e - Math.max(s, cursor)) / spanMs });
    cursor = Math.max(cursor, e);
  }

  return { segments, spanMs };
}

function groupByDate(labels: LabelEntry[]): DateGroup[] {
  const groups: DateGroup[] = [];
  let currentKey = "";
  for (const lbl of labels) {
    const key = dateKey(lbl.start_ts);
    if (key !== currentKey) {
      currentKey = key;
      groups.push({ dateLabel: fmtDate(parseDate(lbl.start_ts)), entries: [], timeline: [], spanMs: 0 });
    }
    groups[groups.length - 1].entries.push(lbl);
  }
  for (const g of groups) {
    const { segments, spanMs } = buildTimeline(g.entries);
    g.timeline = segments;
    g.spanMs = spanMs;
  }
  return groups;
}

const TimelineStrip: Component<{ segments: TimelineSegment[] }> = (props) => {
  const [tooltip, setTooltip] = createSignal<{ text: string; x: number } | null>(null);

  return (
    <div style={{ position: "relative", "margin-top": "3px", "margin-bottom": "4px" }}>
      <div
        style={{
          display: "flex",
          height: "7px",
          "border-radius": "3px",
          overflow: "hidden",
          background: "#1a1a1a",
        }}
      >
        <For each={props.segments}>
          {(seg) => (
            <div
              style={{
                "flex-grow": seg.fraction,
                "flex-basis": "0",
                "min-width": seg.label ? "2px" : "0",
                background: seg.label ? (LABEL_COLORS[seg.label] ?? "#8a8a8a") : "transparent",
                cursor: seg.label ? "pointer" : "default",
                transition: "opacity 0.1s",
              }}
              onMouseEnter={(e) => {
                if (!seg.label) return;
                const rect = (e.currentTarget.parentElement as HTMLElement).getBoundingClientRect();
                const x = e.currentTarget.getBoundingClientRect().left - rect.left + e.currentTarget.offsetWidth / 2;
                const dur = fmtDuration(seg.endMs - seg.startMs);
                const start = fmtTime(new Date(seg.startMs));
                const end = fmtTime(new Date(seg.endMs));
                setTooltip({ text: `${seg.label}  ${start}–${end}  (${dur})`, x });
              }}
              onMouseLeave={() => setTooltip(null)}
            />
          )}
        </For>
      </div>
      <Show when={tooltip()}>
        {(tip) => (
          <div
            style={{
              position: "absolute",
              top: "-22px",
              left: `${Math.max(0, tip().x)}px`,
              transform: "translateX(-50%)",
              background: "#222",
              color: "#e0e0e0",
              padding: "2px 6px",
              "border-radius": "4px",
              "font-size": "0.55rem",
              "white-space": "nowrap",
              "pointer-events": "none",
              "z-index": "10",
              border: "1px solid #333",
            }}
          >
            {tip().text}
          </div>
        )}
      </Show>
    </div>
  );
};

export const LabelHistory: Component<{
  visible: Accessor<boolean>;
}> = (props) => {
  const [labels] = createResource(props.visible, async (show) => {
    if (!show) return [];
    return fetchLabels(50);
  });

  const grouped = createMemo(() => {
    const l = labels();
    return l?.length ? groupByDate(l) : [];
  });

  return (
    <Show when={grouped().length}>
      <div
        style={{
          background: "var(--surface)",
          border: "1px solid #2a2a2a",
          "border-radius": "8px",
          padding: "6px 8px",
          "font-family": "'SF Mono', 'Fira Code', 'Cascadia Code', monospace",
          "font-size": "0.65rem",
          color: "#d0d0d0",
          "box-shadow": "0 8px 32px rgba(0, 0, 0, 0.6)",
        }}
      >
        <div
          style={{
            "font-size": "0.75rem",
            "font-weight": "700",
            color: "#d0d0d0",
            "margin-bottom": "6px",
            "padding-bottom": "4px",
            "border-bottom": "1px solid #2a2a2a",
            "letter-spacing": "0.02em",
          }}
        >
          Label History
        </div>
        <For each={grouped()}>
          {(group) => (
            <div style={{ "margin-bottom": "5px" }}>
              <div
                style={{
                  "font-size": "0.6rem",
                  "font-weight": "700",
                  "text-transform": "uppercase",
                  "letter-spacing": "0.06em",
                  color: "#7a7a7a",
                  "margin-bottom": "1px",
                  "border-bottom": "1px solid #333",
                  "padding-bottom": "1px",
                }}
              >
                {group.dateLabel}
              </div>
              <Show when={group.timeline.length}>
                <TimelineStrip segments={group.timeline} />
              </Show>
              <For each={group.entries}>
                {(lbl) => {
                  const startD = parseDate(lbl.start_ts);
                  const endD = parseDate(lbl.end_ts);
                  const dur = fmtDuration(endD.getTime() - startD.getTime());
                  return (
                    <div
                      style={{
                        display: "flex",
                        "justify-content": "space-between",
                        "align-items": "baseline",
                        padding: "1px 0",
                        gap: "8px",
                      }}
                    >
                      <span
                        style={{
                          display: "flex",
                          "align-items": "center",
                          gap: "6px",
                        }}
                      >
                        <span
                          style={{
                            width: "6px",
                            height: "6px",
                            "border-radius": "50%",
                            background: LABEL_COLORS[lbl.label] ?? "#8a8a8a",
                            "flex-shrink": "0",
                          }}
                        />
                        <span
                          style={{
                            color: LABEL_COLORS[lbl.label] ?? "#d0d0d0",
                            "font-weight": "600",
                            "font-size": "0.65rem",
                          }}
                        >
                          {lbl.label}
                        </span>
                      </span>
                      <span
                        style={{
                          color: "#8a8a8a",
                          "font-size": "0.65rem",
                          "white-space": "nowrap",
                        }}
                      >
                        {fmtTime(startD)} – {fmtTime(endD)}{" "}
                        <span style={{ color: "#5a5a5a" }}>({dur})</span>
                      </span>
                    </div>
                  );
                }}
              </For>
            </div>
          )}
        </For>
      </div>
    </Show>
  );
};
