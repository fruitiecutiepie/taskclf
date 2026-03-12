import type { Component } from "solid-js";

export const StatusProgress: Component<{ pct: number; color?: string }> = (props) => (
  <div
    style={{
      height: "3px",
      background: "#2a2a2a",
      "border-radius": "2px",
      overflow: "hidden",
      "margin-top": "2px",
    }}
  >
    <div
      style={{
        height: "100%",
        width: `${Math.min(100, props.pct)}%`,
        background: props.color ?? "#eab308",
        "border-radius": "2px",
        transition: "width 0.3s ease",
      }}
    />
  </div>
);
