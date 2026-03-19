import type { Component } from "solid-js";

export const StatusRow: Component<{
  label: string;
  value: string;
  color?: string;
  dim?: boolean;
  mono?: boolean;
  tooltip?: string;
}> = (props) => (
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
      title={props.tooltip}
      style={{
        color: "#a0a0a0",
        "font-size": "0.7rem",
        "flex-shrink": "0",
        "user-select": "none",
        cursor: props.tooltip ? "help" : undefined,
        "border-bottom": props.tooltip ? "1px dotted #555" : undefined,
      }}
    >
      {props.label}
    </span>
    <span
      title={props.value}
      style={{
        "font-size": "0.7rem",
        "font-weight": props.dim ? "400" : "600",
        "font-family": props.mono ? "'SF Mono', 'Fira Code', monospace" : "inherit",
        color: props.color ?? (props.dim ? "#a0a0a0" : "#e0e0e0"),
        "text-align": "right",
        overflow: "hidden",
        "text-overflow": "ellipsis",
        "white-space": "nowrap",
        "min-width": "0",
      }}
    >
      {props.value}
    </span>
  </div>
);
