import { type Accessor, type Component, createEffect, createResource, createMemo, createSignal, For, on, Show } from "solid-js";
import { createLabel, deleteLabel, fetchCoreLabels, fetchLabelsByDate, updateLabel } from "../lib/api";
import type { Prediction } from "../lib/ws";
import { ActivityContext, type TimeRange } from "./ActivityContext";
import { FlashMessage } from "./FlashMessage";
import { LABEL_COLORS } from "./StatePanel";

function parseDate(iso: string): Date {
  return new Date(iso);
}

function todayDateStr(): string {
  const d = new Date();
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;
}

function fmtDateLabel(dateStr: string): string {
  const today = todayDateStr();
  if (dateStr === today) return "Today";
  const yesterday = new Date();
  yesterday.setDate(yesterday.getDate() - 1);
  const yStr = `${yesterday.getFullYear()}-${String(yesterday.getMonth() + 1).padStart(2, "0")}-${String(yesterday.getDate()).padStart(2, "0")}`;
  if (dateStr === yStr) return "Yesterday";
  const d = new Date(dateStr + "T00:00:00");
  return d.toLocaleDateString([], { weekday: "short", month: "short", day: "numeric" });
}

function shiftDate(dateStr: string, delta: number): string {
  const d = new Date(dateStr + "T12:00:00");
  d.setDate(d.getDate() + delta);
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;
}

function fmtTime(d: Date): string {
  return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function fmtTimeSec(d: Date): string {
  return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
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

function toTimeInputValue(d: Date): string {
  return `${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}`;
}

function toTimeInputValueSec(d: Date): string {
  return `${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}:${String(d.getSeconds()).padStart(2, "0")}`;
}

function timeInputToDate(dateStr: string, timeVal: string): Date {
  const parts = timeVal.split(":");
  const hh = parts[0] ?? "00";
  const mm = parts[1] ?? "00";
  const ss = parts[2] ?? "00";
  return new Date(`${dateStr}T${hh}:${mm}:${ss}`);
}

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

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

interface GapItem {
  kind: "gap";
  start_ts: string;
  end_ts: string;
}

interface LabelItem {
  kind: "label";
  label: string;
  start_ts: string;
  end_ts: string;
}

type TimelineItem = GapItem | LabelItem;

// ---------------------------------------------------------------------------
// Timeline builder — full-day range
// ---------------------------------------------------------------------------

function buildDayTimeline(
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

// ---------------------------------------------------------------------------
// Item key helper
// ---------------------------------------------------------------------------

function itemKey(item: TimelineItem): string {
  return `${item.kind}|${item.start_ts}|${item.end_ts}`;
}

// ---------------------------------------------------------------------------
// TimelineStrip — with clickable gaps
// ---------------------------------------------------------------------------

const TimelineStrip: Component<{
  segments: TimelineSegment[];
  onSegmentClick?: (seg: TimelineSegment, index: number) => void;
}> = (props) => {
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
          {(seg, idx) => (
            <div
              style={{
                "flex-grow": seg.fraction,
                "flex-basis": "0",
                "min-width": seg.label ? "2px" : seg.fraction > 0.005 ? "1px" : "0",
                background: seg.label
                  ? (LABEL_COLORS[seg.label] ?? "#a0a0a0")
                  : "rgba(255,255,255,0.04)",
                cursor: "pointer",
                transition: "opacity 0.1s, background 0.1s",
              }}
              onMouseEnter={(e) => {
                const rect = (e.currentTarget.parentElement as HTMLElement).getBoundingClientRect();
                const x = e.currentTarget.getBoundingClientRect().left - rect.left + e.currentTarget.offsetWidth / 2;
                const dur = fmtDuration(seg.endMs - seg.startMs);
                const start = fmtTime(new Date(seg.startMs));
                const end = fmtTime(new Date(seg.endMs));
                const prefix = seg.label ?? "Unlabeled";
                setTooltip({ text: `${prefix}  ${start}\u2013${end}  (${dur})`, x });
                if (!seg.label) {
                  e.currentTarget.style.background = "rgba(255,255,255,0.1)";
                }
              }}
              onMouseLeave={(e) => {
                setTooltip(null);
                if (!seg.label) {
                  e.currentTarget.style.background = "rgba(255,255,255,0.04)";
                }
              }}
              onClick={() => {
                if (props.onSegmentClick) {
                  props.onSegmentClick(seg, idx());
                }
              }}
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

// ---------------------------------------------------------------------------
// LabelRow (existing labels)
// ---------------------------------------------------------------------------

const LabelRow: Component<{
  lbl: LabelItem;
  dateStr: string;
  expanded: boolean;
  onToggle: () => void;
  onUpdate: (label: string, newStart: string, newEnd: string) => void;
  onDelete: () => void;
  coreLabels: string[];
  busy: boolean;
  flash: string | null;
}> = (props) => {
  const startD = () => parseDate(props.lbl.start_ts);
  const endD = () => parseDate(props.lbl.end_ts);
  const dur = () => fmtDuration(endD().getTime() - startD().getTime());
  const [confirmDelete, setConfirmDelete] = createSignal(false);
  const [pendingLabel, setPendingLabel] = createSignal<string | null>(null);

  const [startTime, setStartTime] = createSignal(toTimeInputValue(startD()));
  const [endTime, setEndTime] = createSignal(toTimeInputValue(endD()));

  createEffect(() => {
    setStartTime(toTimeInputValue(startD()));
    setEndTime(toTimeInputValue(endD()));
  });

  const editedRange = (): TimeRange | null => {
    const s = timeInputToDate(props.dateStr, startTime());
    const e = timeInputToDate(props.dateStr, endTime());
    if (e.getTime() <= s.getTime()) return null;
    return { start: s.toISOString(), end: e.toISOString() };
  };

  const rangeValid = () => editedRange() !== null;

  const timeChanged = () =>
    startTime() !== toTimeInputValue(startD()) ||
    endTime() !== toTimeInputValue(endD());

  const labelChanged = () => pendingLabel() !== null && pendingLabel() !== props.lbl.label;
  const effectiveLabel = () => pendingLabel() ?? props.lbl.label;
  const hasChanges = () => labelChanged() || timeChanged();

  const timeInputStyle = {
    background: "#111",
    border: "1px solid #333",
    "border-radius": "4px",
    color: "#e0e0e0",
    "font-size": "0.6rem",
    "font-family": "inherit",
    padding: "2px 4px",
    width: "70px",
    "text-align": "center" as const,
  };

  return (
    <div>
      <div
        onClick={props.onToggle}
        style={{
          display: "flex",
          "justify-content": "space-between",
          "align-items": "baseline",
          padding: "2px 4px",
          gap: "8px",
          cursor: "pointer",
          "border-radius": "4px",
          background: props.expanded ? "#1e1e2e" : "transparent",
          transition: "background 0.1s ease",
        }}
        onMouseEnter={(e) => {
          if (!props.expanded) e.currentTarget.style.background = "#1a1a24";
        }}
        onMouseLeave={(e) => {
          if (!props.expanded) e.currentTarget.style.background = "transparent";
        }}
      >
        <span style={{ display: "flex", "align-items": "center", gap: "6px" }}>
          <span
            style={{
              width: "6px",
              height: "6px",
              "border-radius": "50%",
              background: LABEL_COLORS[props.lbl.label] ?? "#a0a0a0",
              "flex-shrink": "0",
            }}
          />
          <span
            style={{
              color: LABEL_COLORS[props.lbl.label] ?? "#e0e0e0",
              "font-weight": "600",
              "font-size": "0.65rem",
            }}
          >
            {props.lbl.label}
          </span>
        </span>
        <span
          style={{ color: "#a0a0a0", "font-size": "0.65rem", "white-space": "nowrap" }}
        >
          {fmtTime(startD())} – {fmtTime(endD())}{" "}
          <span style={{ color: "#808080" }}>({dur()})</span>
        </span>
      </div>

      <Show when={props.expanded}>
        <div
          style={{
            padding: "6px 8px",
            margin: "2px 0 4px",
            background: "#161622",
            "border-radius": "6px",
            border: "1px solid #2a2a3a",
          }}
        >
          <div
            style={{
              display: "flex",
              "align-items": "center",
              "justify-content": "center",
              gap: "6px",
              "margin-bottom": "6px",
            }}
          >
            <span style={{ color: "#888", "font-size": "0.58rem" }}>From</span>
            <input
              type="time"
              value={startTime()}
              max={endTime()}
              onInput={(e) => setStartTime(e.currentTarget.value)}
              style={timeInputStyle}
            />
            <span style={{ color: "#888", "font-size": "0.58rem" }}>to</span>
            <input
              type="time"
              value={endTime()}
              min={startTime()}
              onInput={(e) => setEndTime(e.currentTarget.value)}
              style={timeInputStyle}
            />
            <Show when={timeChanged()}>
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  setStartTime(toTimeInputValue(startD()));
                  setEndTime(toTimeInputValue(endD()));
                }}
                title="Reset time"
                style={{
                  background: "none",
                  border: "none",
                  color: "#888",
                  cursor: "pointer",
                  "font-size": "0.65rem",
                  padding: "0 2px",
                  "line-height": "1",
                  "flex-shrink": "0",
                  transition: "color 0.1s",
                }}
                onMouseEnter={(e) => { e.currentTarget.style.color = "#e0e0e0"; }}
                onMouseLeave={(e) => { e.currentTarget.style.color = "#888"; }}
              >
                ↺
              </button>
            </Show>
          </div>

          <ActivityContext timeRange={() => editedRange()} />

          <Show when={props.flash}>
            <FlashMessage flash={props.flash!} />
          </Show>

          <div
            style={{
              display: "grid",
              "grid-template-columns": "repeat(4, 1fr)",
              gap: "3px",
              "margin-bottom": "6px",
            }}
          >
            <For each={props.coreLabels}>
              {(lbl) => {
                const isOriginal = () => lbl === props.lbl.label;
                const isSelected = () => lbl === effectiveLabel();
                return (
                  <button
                    disabled={props.busy}
                    onClick={(e) => {
                      e.stopPropagation();
                      if (lbl === props.lbl.label) {
                        setPendingLabel(null);
                      } else {
                        setPendingLabel(lbl);
                      }
                    }}
                    style={{
                      padding: "4px 2px",
                      "border-radius": "4px",
                      border: isSelected()
                        ? `1.5px solid ${LABEL_COLORS[lbl] ?? "#888"}`
                        : "1px solid #333",
                      background: isSelected() ? "#1e1e2e" : "#111",
                      color: LABEL_COLORS[lbl] ?? "#e0e0e0",
                      cursor: "pointer",
                      "font-size": "0.58rem",
                      "font-weight": isSelected() ? "700" : "500",
                      "text-align": "center",
                      opacity: props.busy ? "0.5" : isSelected() ? "1" : isOriginal() && labelChanged() ? "0.5" : "0.8",
                      transition: "all 0.1s ease",
                    }}
                  >
                    {lbl}
                  </button>
                );
              }}
            </For>
          </div>

          <Show when={hasChanges() && rangeValid()}>
            <div
              style={{
                "text-align": "center",
                "margin-bottom": "6px",
              }}
            >
              <Show when={labelChanged()}>
                <div style={{ "font-size": "0.58rem", color: "#999", "margin-bottom": "4px" }}>
                  {timeChanged()
                    ? <>Change to <span style={{ color: LABEL_COLORS[effectiveLabel()] ?? "#e0e0e0", "font-weight": "700" }}>{effectiveLabel()}</span> and update time?</>
                    : <>Change to <span style={{ color: LABEL_COLORS[effectiveLabel()] ?? "#e0e0e0", "font-weight": "700" }}>{effectiveLabel()}</span>?</>}
                </div>
              </Show>
              <div
                style={{
                  display: "flex",
                  "justify-content": "center",
                  gap: "6px",
                }}
              >
                <button
                  disabled={props.busy}
                  onClick={(e) => {
                    e.stopPropagation();
                    setPendingLabel(null);
                    setStartTime(toTimeInputValue(startD()));
                    setEndTime(toTimeInputValue(endD()));
                  }}
                  style={{
                    padding: "2px 8px",
                    "border-radius": "4px",
                    border: "1px solid #333",
                    background: "transparent",
                    color: "#999",
                    cursor: "pointer",
                    "font-size": "0.58rem",
                  }}
                >
                  {labelChanged() ? "Cancel" : "Reset"}
                </button>
                <button
                  disabled={props.busy}
                  onClick={(e) => {
                    e.stopPropagation();
                    const r = editedRange();
                    if (r) {
                      const lbl = effectiveLabel();
                      setPendingLabel(null);
                      props.onUpdate(lbl, r.start, r.end);
                    }
                  }}
                  style={{
                    padding: "2px 8px",
                    "border-radius": "4px",
                    border: `1px solid ${LABEL_COLORS[effectiveLabel()] ?? "#888"}`,
                    background: LABEL_COLORS[effectiveLabel()] ?? "#888",
                    color: "#fff",
                    cursor: "pointer",
                    "font-size": "0.58rem",
                    "font-weight": "600",
                  }}
                >
                  {labelChanged() ? "Confirm" : "Update Time"}
                </button>
              </div>
            </div>
          </Show>

          <div
            style={{
              display: "flex",
              "justify-content": "flex-end",
              gap: "6px",
              "padding-top": "4px",
              "border-top": "1px solid #2a2a2a",
            }}
          >
            <Show
              when={confirmDelete()}
              fallback={
                <button
                  disabled={props.busy}
                  onClick={(e) => {
                    e.stopPropagation();
                    setConfirmDelete(true);
                  }}
                  style={{
                    padding: "2px 8px",
                    "border-radius": "4px",
                    border: "1px solid #442222",
                    background: "transparent",
                    color: "#ef4444",
                    cursor: "pointer",
                    "font-size": "0.58rem",
                    opacity: props.busy ? "0.5" : "0.7",
                    transition: "opacity 0.1s",
                  }}
                >
                  Delete
                </button>
              }
            >
              <span style={{ "font-size": "0.58rem", color: "#999", "align-self": "center" }}>
                Delete this label?
              </span>
              <button
                disabled={props.busy}
                onClick={(e) => {
                  e.stopPropagation();
                  setConfirmDelete(false);
                }}
                style={{
                  padding: "2px 8px",
                  "border-radius": "4px",
                  border: "1px solid #333",
                  background: "transparent",
                  color: "#999",
                  cursor: "pointer",
                  "font-size": "0.58rem",
                }}
              >
                Cancel
              </button>
              <button
                disabled={props.busy}
                onClick={(e) => {
                  e.stopPropagation();
                  props.onDelete();
                }}
                style={{
                  padding: "2px 8px",
                  "border-radius": "4px",
                  border: "1px solid #ef4444",
                  background: "#ef4444",
                  color: "#fff",
                  cursor: "pointer",
                  "font-size": "0.58rem",
                  "font-weight": "600",
                }}
              >
                Confirm
              </button>
            </Show>
          </div>
        </div>
      </Show>
    </div>
  );
};

// ---------------------------------------------------------------------------
// GapRow (unlabeled gaps — click to label a sub-range)
// ---------------------------------------------------------------------------

const GapRow: Component<{
  gap: GapItem;
  dateStr: string;
  expanded: boolean;
  onToggle: () => void;
  onCreate: (start: string, end: string, label: string) => void;
  coreLabels: string[];
  busy: boolean;
  flash: string | null;
}> = (props) => {
  const gapStartD = () => parseDate(props.gap.start_ts);
  const gapEndD = () => parseDate(props.gap.end_ts);
  const dur = () => fmtDuration(gapEndD().getTime() - gapStartD().getTime());

  const [startIso, setStartIso] = createSignal(props.gap.start_ts);
  const [endIso, setEndIso] = createSignal(props.gap.end_ts);

  const [startTimeDisplay, setStartTimeDisplay] = createSignal(toTimeInputValueSec(gapStartD()));
  const [endTimeDisplay, setEndTimeDisplay] = createSignal(toTimeInputValueSec(gapEndD()));

  createEffect(() => {
    setStartIso(props.gap.start_ts);
    setEndIso(props.gap.end_ts);
    setStartTimeDisplay(toTimeInputValueSec(gapStartD()));
    setEndTimeDisplay(toTimeInputValueSec(gapEndD()));
  });

  function handleStartChange(val: string) {
    setStartTimeDisplay(val);
    setStartIso(timeInputToDate(props.dateStr, val).toISOString());
  }

  function handleEndChange(val: string) {
    setEndTimeDisplay(val);
    setEndIso(timeInputToDate(props.dateStr, val).toISOString());
  }

  const selectedRange = (): TimeRange | null => {
    const s = parseDate(startIso()).getTime();
    const e = parseDate(endIso()).getTime();
    if (e <= s) return null;
    return { start: startIso(), end: endIso() };
  };

  const rangeValid = () => selectedRange() !== null;

  const timeInputStyle = {
    background: "#111",
    border: "1px solid #333",
    "border-radius": "4px",
    color: "#e0e0e0",
    "font-size": "0.6rem",
    "font-family": "inherit",
    padding: "2px 4px",
    width: "78px",
    "text-align": "center" as const,
  };

  return (
    <div>
      <div
        onClick={props.onToggle}
        style={{
          display: "flex",
          "justify-content": "space-between",
          "align-items": "baseline",
          padding: "2px 4px",
          gap: "8px",
          cursor: "pointer",
          "border-radius": "4px",
          background: props.expanded ? "#1a1a22" : "transparent",
          transition: "background 0.1s ease",
        }}
        onMouseEnter={(e) => {
          if (!props.expanded) e.currentTarget.style.background = "#16161e";
        }}
        onMouseLeave={(e) => {
          if (!props.expanded) e.currentTarget.style.background = "transparent";
        }}
      >
        <span style={{ display: "flex", "align-items": "center", gap: "6px" }}>
          <span
            style={{
              width: "6px",
              height: "6px",
              "border-radius": "50%",
              border: "1.5px dashed #555",
              "flex-shrink": "0",
            }}
          />
          <span
            style={{
              color: "#666",
              "font-weight": "500",
              "font-size": "0.65rem",
              "font-style": "italic",
            }}
          >
            Unlabeled
          </span>
        </span>
        <span
          style={{ color: "#666", "font-size": "0.65rem", "white-space": "nowrap" }}
        >
          {fmtTimeSec(gapStartD())} – {fmtTimeSec(gapEndD())}{" "}
          <span style={{ color: "#555" }}>({dur()})</span>
        </span>
      </div>

      <Show when={props.expanded}>
        <div
          style={{
            padding: "6px 8px",
            margin: "2px 0 4px",
            background: "#131320",
            "border-radius": "6px",
            border: "1px dashed #2a2a3a",
          }}
        >
          <div
            style={{
              display: "flex",
              "align-items": "center",
              "justify-content": "center",
              gap: "6px",
              "margin-bottom": "6px",
            }}
          >
            <span style={{ color: "#888", "font-size": "0.58rem" }}>From</span>
            <input
              type="time"
              step="1"
              value={startTimeDisplay()}
              max={endTimeDisplay()}
              onInput={(e) => handleStartChange(e.currentTarget.value)}
              style={timeInputStyle}
            />
            <span style={{ color: "#888", "font-size": "0.58rem" }}>to</span>
            <input
              type="time"
              step="1"
              value={endTimeDisplay()}
              min={startTimeDisplay()}
              onInput={(e) => handleEndChange(e.currentTarget.value)}
              style={timeInputStyle}
            />
          </div>

          <ActivityContext timeRange={() => selectedRange()} />

          <Show when={props.flash}>
            <FlashMessage flash={props.flash!} />
          </Show>

          <div
            style={{
              display: "grid",
              "grid-template-columns": "repeat(4, 1fr)",
              gap: "3px",
            }}
          >
            <For each={props.coreLabels}>
              {(lbl) => (
                <button
                  disabled={props.busy || !rangeValid()}
                  onClick={(e) => {
                    e.stopPropagation();
                    const r = selectedRange();
                    if (r) props.onCreate(r.start, r.end, lbl);
                  }}
                  style={{
                    padding: "4px 2px",
                    "border-radius": "4px",
                    border: "1px solid #333",
                    background: "#111",
                    color: LABEL_COLORS[lbl] ?? "#e0e0e0",
                    cursor: rangeValid() ? "pointer" : "not-allowed",
                    "font-size": "0.58rem",
                    "font-weight": "500",
                    "text-align": "center",
                    opacity: props.busy || !rangeValid() ? "0.4" : "0.8",
                    transition: "all 0.1s ease",
                  }}
                >
                  {lbl}
                </button>
              )}
            </For>
          </div>
        </div>
      </Show>
    </div>
  );
};

// ---------------------------------------------------------------------------
// LabelHistory (main export)
// ---------------------------------------------------------------------------

export const LabelHistory: Component<{
  visible: Accessor<boolean>;
  latestPrediction?: Accessor<Prediction | null>;
}> = (props) => {
  const [selectedDate, setSelectedDate] = createSignal(todayDateStr());
  let dateInputRef: HTMLInputElement | undefined;

  const [labels, { refetch }] = createResource(
    () => (props.visible() ? selectedDate() : null),
    async (dateStr) => {
      if (!dateStr) return [];
      return fetchLabelsByDate(dateStr);
    },
  );
  const [coreLabels] = createResource(fetchCoreLabels);

  const [expandedKey, setExpandedKey] = createSignal<string | null>(null);
  const [busy, setBusy] = createSignal(false);
  const [flash, setFlash] = createSignal<string | null>(null);

  createEffect(on(
    () => props.latestPrediction?.(),
    () => { if (props.visible()) refetch(); },
    { defer: true },
  ));

  const dayData = createMemo(() => {
    const l = labels();
    const entries: LabelEntry[] = (l ?? []).map((r) => ({
      label: r.label,
      start_ts: r.start_ts,
      end_ts: r.end_ts,
    }));
    return buildDayTimeline(entries, selectedDate());
  });

  function toggleRow(item: TimelineItem) {
    const key = itemKey(item);
    setExpandedKey(expandedKey() === key ? null : key);
    setFlash(null);
  }

  async function handleUpdate(item: LabelItem, newLabel: string, newStart: string, newEnd: string) {
    setBusy(true);
    setFlash(null);
    try {
      await updateLabel({
        start_ts: item.start_ts,
        end_ts: item.end_ts,
        label: newLabel,
        new_start_ts: newStart,
        new_end_ts: newEnd,
      });
      setFlash(newLabel);
      setTimeout(() => {
        setFlash(null);
        setExpandedKey(null);
        refetch();
      }, 800);
    } catch (err: any) {
      setFlash(`Error: ${err.message}`);
    } finally {
      setBusy(false);
    }
  }

  async function handleDelete(item: LabelItem) {
    setBusy(true);
    setFlash(null);
    try {
      await deleteLabel({
        start_ts: item.start_ts,
        end_ts: item.end_ts,
      });
      setExpandedKey(null);
      refetch();
    } catch (err: any) {
      setFlash(`Error: ${err.message}`);
    } finally {
      setBusy(false);
    }
  }

  async function handleGapCreate(startTs: string, endTs: string, label: string) {
    setBusy(true);
    setFlash(null);
    try {
      await createLabel({
        start_ts: startTs,
        end_ts: endTs,
        label,
      });
      setFlash(label);
      setTimeout(() => {
        setFlash(null);
        setExpandedKey(null);
        refetch();
      }, 800);
    } catch (err: any) {
      setFlash(`Error: ${err.message}`);
    } finally {
      setBusy(false);
    }
  }

  const isFutureDate = () => selectedDate() >= shiftDate(todayDateStr(), 1);

  return (
    <div
      style={{
        padding: "6px 8px",
        "font-family": "'SF Mono', 'Fira Code', 'Cascadia Code', monospace",
        "font-size": "0.65rem",
        color: "#e0e0e0",
      }}
    >
      {/* Header with date navigation */}
      <div
        style={{
          display: "flex",
          "align-items": "center",
          "justify-content": "space-between",
          "margin-bottom": "6px",
          "padding-bottom": "4px",
          "border-bottom": "1px solid #2a2a2a",
        }}
      >
        <button
          onClick={() => setSelectedDate(shiftDate(selectedDate(), -1))}
          style={{
            background: "none",
            border: "none",
            color: "#999",
            cursor: "pointer",
            "font-size": "0.75rem",
            padding: "2px 6px",
            "border-radius": "4px",
            transition: "color 0.1s",
          }}
          onMouseEnter={(e) => { e.currentTarget.style.color = "#e0e0e0"; }}
          onMouseLeave={(e) => { e.currentTarget.style.color = "#999"; }}
        >
          ◀
        </button>
        <div style={{ position: "relative" }}>
          <span
            onClick={() => dateInputRef?.showPicker()}
            style={{
              "font-size": "0.75rem",
              "font-weight": "700",
              color: "#e0e0e0",
              "letter-spacing": "0.02em",
              cursor: "pointer",
              "user-select": "none",
            }}
          >
            {fmtDateLabel(selectedDate())}
          </span>
          <input
            ref={dateInputRef}
            type="date"
            value={selectedDate()}
            max={todayDateStr()}
            onInput={(e) => {
              const v = e.currentTarget.value;
              if (v) setSelectedDate(v);
            }}
            style={{
              position: "absolute",
              top: "0",
              left: "0",
              width: "100%",
              height: "100%",
              opacity: "0",
              cursor: "pointer",
            }}
          />
        </div>
        <button
          onClick={() => { if (!isFutureDate()) setSelectedDate(shiftDate(selectedDate(), 1)); }}
          disabled={isFutureDate()}
          style={{
            background: "none",
            border: "none",
            color: isFutureDate() ? "#444" : "#999",
            cursor: isFutureDate() ? "default" : "pointer",
            "font-size": "0.75rem",
            padding: "2px 6px",
            "border-radius": "4px",
            transition: "color 0.1s",
          }}
          onMouseEnter={(e) => { if (!isFutureDate()) e.currentTarget.style.color = "#e0e0e0"; }}
          onMouseLeave={(e) => { if (!isFutureDate()) e.currentTarget.style.color = "#999"; }}
        >
          ▶
        </button>
      </div>

      {/* Content */}
      <Show
        when={!labels.loading}
        fallback={
          <div
            style={{
              "text-align": "center",
              padding: "16px 8px",
              color: "#707070",
              "font-size": "0.6rem",
            }}
          >
            Loading labels…
          </div>
        }
      >
        <TimelineStrip
          segments={dayData().segments}
          onSegmentClick={(_seg, index) => {
            const item = dayData().items[index];
            if (!item) return;
            const key = itemKey(item);
            setExpandedKey(expandedKey() === key ? null : key);
            setFlash(null);
          }}
        />
        <For each={dayData().items}>
          {(item) => (
            <Show
              when={item.kind === "label"}
              fallback={
                <GapRow
                  gap={item as GapItem}
                  dateStr={selectedDate()}
                  expanded={expandedKey() === itemKey(item)}
                  onToggle={() => toggleRow(item)}
                  onCreate={handleGapCreate}
                  coreLabels={coreLabels() ?? []}
                  busy={busy()}
                  flash={expandedKey() === itemKey(item) ? flash() : null}
                />
              }
            >
              <LabelRow
                lbl={item as LabelItem}
                dateStr={selectedDate()}
                expanded={expandedKey() === itemKey(item)}
                onToggle={() => toggleRow(item)}
                onUpdate={(newLabel, newStart, newEnd) => handleUpdate(item as LabelItem, newLabel, newStart, newEnd)}
                onDelete={() => handleDelete(item as LabelItem)}
                coreLabels={coreLabels() ?? []}
                busy={busy()}
                flash={expandedKey() === itemKey(item) ? flash() : null}
              />
            </Show>
          )}
        </For>
      </Show>
    </div>
  );
};
