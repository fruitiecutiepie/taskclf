import { type Component, createEffect, createSignal, For, Show } from "solid-js";
import type { TimeRange } from "../lib/date";
import {
  fmtDuration,
  fmtTime,
  parseDate,
  timeInputToDate,
  toTimeInputValue,
} from "../lib/date";
import { LABEL_COLORS } from "../lib/labelColors";
import type { LabelItem } from "../lib/labelTimeline";
import { ActivitySummary } from "./ActivitySummary";
import { LabelFlash } from "./LabelFlash";

export const LabelHistoryRow: Component<{
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

  const labelChanged = () =>
    pendingLabel() !== null && pendingLabel() !== props.lbl.label;
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
          background: props.expanded ? "#1e1e2e" : "transparent",
          transition: "background 0.1s ease",
          border: "none",
          font: "inherit",
          color: "inherit",
          width: "100%",
          "text-align": "left",
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
                type="button"
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

          <ActivitySummary timeRange={() => editedRange()} />

          <Show when={props.flash}>
            <LabelFlash flash={props.flash as string} />
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
                    type="button"
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
                      opacity: props.busy
                        ? "0.5"
                        : isSelected()
                          ? "1"
                          : isOriginal() && labelChanged()
                            ? "0.5"
                            : "0.8",
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
                <div
                  style={{
                    "font-size": "0.58rem",
                    color: "#999",
                    "margin-bottom": "4px",
                  }}
                >
                  {timeChanged() ? (
                    <>
                      Change to{" "}
                      <span
                        style={{
                          color: LABEL_COLORS[effectiveLabel()] ?? "#e0e0e0",
                          "font-weight": "700",
                        }}
                      >
                        {effectiveLabel()}
                      </span>{" "}
                      and update time?
                    </>
                  ) : (
                    <>
                      Change to{" "}
                      <span
                        style={{
                          color: LABEL_COLORS[effectiveLabel()] ?? "#e0e0e0",
                          "font-weight": "700",
                        }}
                      >
                        {effectiveLabel()}
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
                  type="button"
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
                  type="button"
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
                type="button"
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
