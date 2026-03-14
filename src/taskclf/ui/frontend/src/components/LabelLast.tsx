import { type Accessor, type Component, Show } from "solid-js";
import { timeAgo } from "../lib/date";
import { LABEL_COLORS } from "../lib/labelColors";

interface LabelLastProps {
  lastLabel: Accessor<{ label: string; end_ts: string } | null | undefined>;
}

export const LabelLast: Component<LabelLastProps> = (props) => (
  <div
    style={{
      "text-align": "center",
      "font-size": "0.65rem",
      color: "var(--text-muted)",
      "margin-top": "6px",
      "margin-bottom": "2px",
      "padding-top": "4px",
      "border-top": "1px solid var(--border)",
    }}
  >
    <Show
      when={props.lastLabel()}
      fallback={<span style={{ color: "var(--text-muted)" }}>No labels yet</span>}
    >
      Last:{" "}
      <span
        style={{
          color: LABEL_COLORS[props.lastLabel()?.label] ?? "var(--text)",
          "font-weight": "600",
        }}
      >
        {props.lastLabel()?.label}
      </span>{" "}
      {timeAgo(props.lastLabel()?.end_ts)}
    </Show>
  </div>
);
