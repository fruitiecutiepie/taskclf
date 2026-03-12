import { For, type Component } from "solid-js";
import { parseISODate } from "../lib/date";
import { LABEL_COLORS } from "../lib/labelColors";

export interface OverwritePending {
  label: string;
  start: string;
  end: string;
  conflicts: { start_ts: string; end_ts: string; label: string }[];
  confidence: number;
  extendForward: boolean;
}

interface LabelOverwriteProps {
  pending: OverwritePending;
  onConfirm: () => void;
  onCancel: () => void;
}

const fmt = (d: Date) =>
  d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });

export const LabelOverwrite: Component<LabelOverwriteProps> = (props) => (
  <div
    style={{
      "text-align": "center",
      "font-size": "0.8rem",
      "margin-top": "8px",
      "margin-bottom": "8px",
      color: "var(--danger)",
    }}
  >
    <span>
      Overlaps{" "}
      <For each={props.pending.conflicts}>
        {(c, i) => {
          const color = LABEL_COLORS[c.label] ?? "var(--text)";
          const cs = parseISODate(c.start_ts);
          const ce = parseISODate(c.end_ts);
          return (
            <>
              {i() > 0 && ", "}
              <span style={{ color, "font-weight": "700" }}>{c.label}</span>
              {" "}{fmt(cs)}{"\u2013"}{fmt(ce)}
            </>
          );
        }}
      </For>
      . Overwrite?
    </span>
    <div
      style={{
        display: "flex",
        "justify-content": "center",
        gap: "8px",
        "margin-top": "4px",
      }}
    >
      <button
        onClick={props.onConfirm}
        style={{
          padding: "2px 10px",
          "border-radius": "6px",
          border: "1px solid var(--danger)",
          background: "var(--danger)",
          color: "#fff",
          cursor: "pointer",
          "font-size": "0.7rem",
          "font-weight": "600",
        }}
      >
        Yes
      </button>
      <button
        onClick={props.onCancel}
        style={{
          padding: "2px 10px",
          "border-radius": "6px",
          border: "1px solid var(--border)",
          background: "var(--surface)",
          color: "var(--text-muted)",
          cursor: "pointer",
          "font-size": "0.7rem",
          "font-weight": "600",
        }}
      >
        No
      </button>
    </div>
  </div>
);
