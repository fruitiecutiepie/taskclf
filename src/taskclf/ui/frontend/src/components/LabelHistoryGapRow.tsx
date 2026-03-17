import { type Component, createEffect, createSignal, For, Show } from "solid-js";
import type { TimeRange } from "../lib/date";
import {
  fmtDuration,
  fmtTimeSec,
  parseDate,
  timeInputToDate,
  toTimeInputValueSec,
} from "../lib/date";
import { LABEL_COLORS } from "../lib/labelColors";
import type { GapItem } from "../lib/labelTimeline";
import { ActivitySummary } from "./ActivitySummary";
import { LabelFlash } from "./LabelFlash";

export const LabelHistoryGapRow: Component<{
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

  const [startTimeDisplay, setStartTimeDisplay] = createSignal(
    toTimeInputValueSec(gapStartD()),
  );
  const [endTimeDisplay, setEndTimeDisplay] = createSignal(
    toTimeInputValueSec(gapEndD()),
  );

  createEffect(() => {
    setStartIso(props.gap.start_ts);
    setEndIso(props.gap.end_ts);
    setStartTimeDisplay(toTimeInputValueSec(gapStartD()));
    setEndTimeDisplay(toTimeInputValueSec(gapEndD()));
  });

  function gap_start_time_set(val: string) {
    setStartTimeDisplay(val);
    setStartIso(timeInputToDate(props.dateStr, val).toISOString());
  }

  function gap_end_time_set(val: string) {
    setEndTimeDisplay(val);
    setEndIso(timeInputToDate(props.dateStr, val).toISOString());
  }

  const selectedRange = (): TimeRange | null => {
    const s = parseDate(startIso()).getTime();
    const e = parseDate(endIso()).getTime();
    if (e <= s) {
      return null;
    }
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
      <button
        type="button"
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
          border: "none",
          font: "inherit",
          color: "inherit",
          width: "100%",
          "text-align": "left",
        }}
        onMouseEnter={(e) => {
          if (!props.expanded) {
            e.currentTarget.style.background = "#16161e";
          }
        }}
        onMouseLeave={(e) => {
          if (!props.expanded) {
            e.currentTarget.style.background = "transparent";
          }
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
      </button>

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
              onInput={(e) => gap_start_time_set(e.currentTarget.value)}
              style={timeInputStyle}
            />
            <span style={{ color: "#888", "font-size": "0.58rem" }}>to</span>
            <input
              type="time"
              step="1"
              value={endTimeDisplay()}
              min={startTimeDisplay()}
              onInput={(e) => gap_end_time_set(e.currentTarget.value)}
              style={timeInputStyle}
            />
          </div>

          <ActivitySummary timeRange={() => selectedRange()} />

          {props.flash && <LabelFlash flash={props.flash} />}

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
                  type="button"
                  disabled={props.busy || !rangeValid()}
                  onClick={(e) => {
                    e.stopPropagation();
                    const r = selectedRange();
                    if (r) {
                      props.onCreate(r.start, r.end, lbl);
                    }
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
