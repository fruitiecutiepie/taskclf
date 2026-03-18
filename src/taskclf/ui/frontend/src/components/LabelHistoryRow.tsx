import { type Component, createEffect, createSignal, For, Show } from "solid-js";
import type { TimeRange } from "../lib/date";
import {
  date_parse,
  duration_fmt,
  time_fmt,
  time_input_date,
  time_input_value,
} from "../lib/date";
import { LABEL_COLORS } from "../lib/labelColors";
import type { LabelItem } from "../lib/labelTimeline";
import { ActivitySummary } from "./ActivitySummary";
import { LabelFlash } from "./LabelFlash";

export const LabelHistoryRow: Component<{
  label_item: LabelItem;
  date_str: string;
  expanded: boolean;
  on_toggle: () => void;
  on_update: (label: string, new_start: string, new_end: string) => void;
  on_delete: () => void;
  core_labels: string[];
  busy: boolean;
  flash: string | null;
}> = (props) => {
  const is_open_ended = () => props.label_item.open_ended === true;
  const start_d = () => date_parse(props.label_item.start_ts);
  const end_d = () => date_parse(props.label_item.end_ts);
  const duration = () =>
    is_open_ended()
      ? "until next label"
      : duration_fmt(end_d().getTime() - start_d().getTime());
  const [confirm_delete, set_confirm_delete] = createSignal(false);
  const [pending_label, set_pending_label] = createSignal<string | null>(null);

  const [start_time, set_start_time] = createSignal(time_input_value(start_d()));
  const [end_time, set_end_time] = createSignal(time_input_value(end_d()));

  createEffect(() => {
    set_start_time(time_input_value(start_d()));
    set_end_time(time_input_value(end_d()));
  });

  const edited_range = (): TimeRange | null => {
    const start = time_input_date(props.date_str, start_time());
    const end = time_input_date(props.date_str, end_time());
    if (end.getTime() <= start.getTime()) {
      return null;
    }
    return { start: start.toISOString(), end: end.toISOString() };
  };

  const range_valid = () => edited_range() !== null;

  const time_changed = () =>
    start_time() !== time_input_value(start_d())
    || end_time() !== time_input_value(end_d());

  const label_changed = () =>
    pending_label() !== null && pending_label() !== props.label_item.label;
  const effective_label = () => pending_label() ?? props.label_item.label;
  const has_changes = () => label_changed() || time_changed();

  const time_input_style = {
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
          background: props.expanded ? "#1e1e2e" : "transparent",
          transition: "background 0.1s ease",
          border: "none",
          font: "inherit",
          color: "inherit",
          width: "100%",
          "text-align": "left",
        }}
        onMouseEnter={(e) => {
          if (!props.expanded) {
            e.currentTarget.style.background = "#1a1a24";
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
              background: LABEL_COLORS[props.label_item.label] ?? "#a0a0a0",
              "flex-shrink": "0",
            }}
          />
          <span
            style={{
              color: LABEL_COLORS[props.label_item.label] ?? "#e0e0e0",
              "font-weight": "600",
              "font-size": "0.65rem",
            }}
          >
            {props.label_item.label}
          </span>
        </span>
        <span
          style={{ color: "#a0a0a0", "font-size": "0.65rem", "white-space": "nowrap" }}
        >
          {time_fmt(start_d())} – {is_open_ended() ? "Now" : time_fmt(end_d())}{" "}
          <span style={{ color: "#808080" }}>({duration()})</span>
        </span>
      </button>

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
              value={start_time()}
              max={end_time()}
              onInput={(e) => set_start_time(e.currentTarget.value)}
              style={time_input_style}
            />
            <span style={{ color: "#888", "font-size": "0.58rem" }}>to</span>
            <input
              type="time"
              value={end_time()}
              min={start_time()}
              onInput={(e) => set_end_time(e.currentTarget.value)}
              style={time_input_style}
            />
            <Show when={time_changed()}>
              <button
                type="button"
                onClick={(e) => {
                  e.stopPropagation();
                  set_start_time(time_input_value(start_d()));
                  set_end_time(time_input_value(end_d()));
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
                onMouseEnter={(e) => {
                  e.currentTarget.style.color = "#e0e0e0";
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.color = "#888";
                }}
              >
                ↺
              </button>
            </Show>
          </div>

          <ActivitySummary time_range={() => edited_range()} />

          {props.flash && <LabelFlash flash={props.flash} />}

          <div
            style={{
              display: "grid",
              "grid-template-columns": "repeat(4, 1fr)",
              gap: "3px",
              "margin-bottom": "6px",
            }}
          >
            <For each={props.core_labels}>
              {(label_name) => {
                const is_original = () => label_name === props.label_item.label;
                const is_selected = () => label_name === effective_label();
                return (
                  <button
                    type="button"
                    disabled={props.busy}
                    onClick={(e) => {
                      e.stopPropagation();
                      if (label_name === props.label_item.label) {
                        set_pending_label(null);
                      } else {
                        set_pending_label(label_name);
                      }
                    }}
                    style={{
                      padding: "4px 2px",
                      "border-radius": "4px",
                      border: is_selected()
                        ? `1.5px solid ${LABEL_COLORS[label_name] ?? "#888"}`
                        : "1px solid #333",
                      background: is_selected() ? "#1e1e2e" : "#111",
                      color: LABEL_COLORS[label_name] ?? "#e0e0e0",
                      cursor: "pointer",
                      "font-size": "0.58rem",
                      "font-weight": is_selected() ? "700" : "500",
                      "text-align": "center",
                      opacity: props.busy
                        ? "0.5"
                        : is_selected()
                          ? "1"
                          : is_original() && label_changed()
                            ? "0.5"
                            : "0.8",
                      transition: "all 0.1s ease",
                    }}
                  >
                    {label_name}
                  </button>
                );
              }}
            </For>
          </div>

          <Show when={has_changes() && range_valid()}>
            <div
              style={{
                "text-align": "center",
                "margin-bottom": "6px",
              }}
            >
              <Show when={label_changed()}>
                <div
                  style={{
                    "font-size": "0.58rem",
                    color: "#999",
                    "margin-bottom": "4px",
                  }}
                >
                  {time_changed() ? (
                    <>
                      Change to{" "}
                      <span
                        style={{
                          color: LABEL_COLORS[effective_label()] ?? "#e0e0e0",
                          "font-weight": "700",
                        }}
                      >
                        {effective_label()}
                      </span>{" "}
                      and update time?
                    </>
                  ) : (
                    <>
                      Change to{" "}
                      <span
                        style={{
                          color: LABEL_COLORS[effective_label()] ?? "#e0e0e0",
                          "font-weight": "700",
                        }}
                      >
                        {effective_label()}
                      </span>
                      ?
                    </>
                  )}
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
                  type="button"
                  disabled={props.busy}
                  onClick={(e) => {
                    e.stopPropagation();
                    set_pending_label(null);
                    set_start_time(time_input_value(start_d()));
                    set_end_time(time_input_value(end_d()));
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
                  {label_changed() ? "Cancel" : "Reset"}
                </button>
                <button
                  type="button"
                  disabled={props.busy}
                  onClick={(e) => {
                    e.stopPropagation();
                    const r = edited_range();
                    if (r) {
                      const label = effective_label();
                      set_pending_label(null);
                      props.on_update(label, r.start, r.end);
                    }
                  }}
                  style={{
                    padding: "2px 8px",
                    "border-radius": "4px",
                    border: `1px solid ${LABEL_COLORS[effective_label()] ?? "#888"}`,
                    background: LABEL_COLORS[effective_label()] ?? "#888",
                    color: "#fff",
                    cursor: "pointer",
                    "font-size": "0.58rem",
                    "font-weight": "600",
                  }}
                >
                  {label_changed() ? "Confirm" : "Update Time"}
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
              when={confirm_delete()}
              fallback={
                <button
                  type="button"
                  disabled={props.busy}
                  onClick={(e) => {
                    e.stopPropagation();
                    set_confirm_delete(true);
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
              <span
                style={{
                  "font-size": "0.58rem",
                  color: "#999",
                  "align-self": "center",
                }}
              >
                Delete this label?
              </span>
              <button
                type="button"
                disabled={props.busy}
                onClick={(e) => {
                  e.stopPropagation();
                  set_confirm_delete(false);
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
                type="button"
                disabled={props.busy}
                onClick={(e) => {
                  e.stopPropagation();
                  props.on_delete();
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
