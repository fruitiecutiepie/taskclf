import type { Component } from "solid-js";
import { Show } from "solid-js";
import { LABEL_COLORS } from "../lib/labelColors";

type LabelFlashProps = {
  flash: string;
};

export const LabelFlash: Component<LabelFlashProps> = (props) => {
  const is_error = () => props.flash.startsWith("Error");
  const label_color = () => LABEL_COLORS[props.flash] ?? "#22c55e";

  return (
    <div
      style={{
        display: "flex",
        "align-items": "center",
        "justify-content": "center",
        gap: "5px",
        "margin-bottom": "6px",
        padding: "4px 10px",
        "border-radius": "5px",
        background: is_error() ? "rgba(239,68,68,0.08)" : "rgba(34,197,94,0.06)",
        border: is_error()
          ? "1px solid rgba(239,68,68,0.2)"
          : "1px solid rgba(34,197,94,0.15)",
      }}
    >
      <span
        style={{
          "font-size": "0.6rem",
          color: is_error() ? "#ef4444" : "#22c55e",
          "line-height": "1",
        }}
      >
        {is_error() ? "✕" : "✓"}
      </span>
      <Show
        when={!is_error()}
        fallback={
          <span style={{ "font-size": "0.58rem", color: "#ef4444" }}>
            {props.flash}
          </span>
        }
      >
        <span style={{ "font-size": "0.58rem", color: "#888" }}>Saved as</span>
        <span
          style={{
            "font-size": "0.58rem",
            "font-weight": "700",
            color: label_color(),
          }}
        >
          {props.flash}
        </span>
      </Show>
    </div>
  );
};
