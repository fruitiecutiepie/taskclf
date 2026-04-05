import { type Component, createEffect, createSignal, For, Show } from "solid-js";
import type { TimeRange } from "../lib/date";
import {
  date_parse,
  duration_fmt,
  time_input_date,
  time_input_value_sec,
  time_sec_fmt,
} from "../lib/date";
import { LABEL_COLORS } from "../lib/labelColors";
import type { GapItem } from "../lib/labelTimeline";
import { ActivitySummary } from "./ActivitySummary";
import { ErrorBanner } from "./ErrorBanner";
import { LabelFlash } from "./LabelFlash";

export const LabelHistoryGapRow: Component<{
  gap: GapItem;
  date_str: string;
  expanded: boolean;
  on_toggle: () => void;
  on_create: (start: string, end: string, label: string) => void;
  core_labels: string[];
  busy: boolean;
  flash: string | null;
  error: string | null;
  on_error_close: () => void;
}> = (props) => {
  const gap_start_d = () => date_parse(props.gap.start_ts);
  const gap_end_d = () => date_parse(props.gap.end_ts);
  const duration = () => duration_fmt(gap_end_d().getTime() - gap_start_d().getTime());

  const [start_iso, set_start_iso] = createSignal(props.gap.start_ts);
  const [end_iso, set_end_iso] = createSignal(props.gap.end_ts);

  const [start_time_display, set_start_time_display] = createSignal(
    time_input_value_sec(gap_start_d()),
  );
  const [end_time_display, set_end_time_display] = createSignal(
    time_input_value_sec(gap_end_d()),
  );

  createEffect(() => {
    set_start_iso(props.gap.start_ts);
    set_end_iso(props.gap.end_ts);
    set_start_time_display(time_input_value_sec(gap_start_d()));
    set_end_time_display(time_input_value_sec(gap_end_d()));
  });

  function gap_start_time_set(val: string) {
    set_start_time_display(val);
    set_start_iso(time_input_date(props.date_str, val).toISOString());
  }

  function gap_end_time_set(val: string) {
    set_end_time_display(val);
    set_end_iso(time_input_date(props.date_str, val).toISOString());
  }

  const selected_range = (): TimeRange | null => {
    const s = date_parse(start_iso()).getTime();
    const e = date_parse(end_iso()).getTime();
    if (e <= s) {
      return null;
    }
    return { start: start_iso(), end: end_iso() };
  };

  const range_valid = () => selected_range() !== null;

  const time_input_style = {
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
        onClick={props.on_toggle}
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
          {time_sec_fmt(gap_start_d())} – {time_sec_fmt(gap_end_d())}{" "}
          <span style={{ color: "#555" }}>({duration()})</span>
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
              value={start_time_display()}
              max={end_time_display()}
              onInput={(e) => gap_start_time_set(e.currentTarget.value)}
              style={time_input_style}
            />
            <span style={{ color: "#888", "font-size": "0.58rem" }}>to</span>
            <input
              type="time"
              step="1"
              value={end_time_display()}
              min={start_time_display()}
              onInput={(e) => gap_end_time_set(e.currentTarget.value)}
              style={time_input_style}
            />
          </div>

          <ActivitySummary time_range={() => selected_range()} />

          <Show when={props.error}>
            <ErrorBanner message={props.error ?? ""} on_close={props.on_error_close} />
          </Show>

          {props.flash && <LabelFlash flash={props.flash} />}

          <div
            style={{
              display: "grid",
              "grid-template-columns": "repeat(4, 1fr)",
              gap: "3px",
            }}
          >
            <For each={props.core_labels}>
              {(lbl) => (
                <button
                  type="button"
                  disabled={props.busy || !range_valid()}
                  onClick={(e) => {
                    e.stopPropagation();
                    const r = selected_range();
                    if (r) {
                      props.on_create(r.start, r.end, lbl);
                    }
                  }}
                  style={{
                    padding: "4px 2px",
                    "border-radius": "4px",
                    border: "1px solid #333",
                    background: "#111",
                    color: LABEL_COLORS[lbl] ?? "#e0e0e0",
                    cursor: range_valid() ? "pointer" : "not-allowed",
                    "font-size": "0.58rem",
                    "font-weight": "500",
                    "text-align": "center",
                    opacity: props.busy || !range_valid() ? "0.4" : "0.8",
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
