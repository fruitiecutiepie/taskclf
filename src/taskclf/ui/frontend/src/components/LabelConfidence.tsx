import type { Accessor, Component } from "solid-js";

interface LabelConfidenceProps {
  value: Accessor<number>;
  onChange: (v: number) => void;
}

export const LabelConfidence: Component<LabelConfidenceProps> = (props) => (
  <div
    style={{
      display: "flex",
      "align-items": "center",
      "justify-content": "center",
      gap: "6px",
      "margin-bottom": "6px",
    }}
  >
    <span
      style={{
        "font-size": "0.7rem",
        color: "var(--text-muted)",
        "flex-shrink": "0",
      }}
    >
      Confidence
    </span>
    <input
      type="range"
      min="0"
      max="100"
      step="5"
      value={props.value()}
      onInput={(e) => props.onChange(parseInt(e.currentTarget.value))}
      style={{
        flex: "1",
        height: "4px",
        "max-width": "120px",
        "accent-color": "var(--accent)",
        cursor: "pointer",
      }}
    />
    <span
      style={{
        "font-size": "0.7rem",
        "font-weight": props.value() < 100 ? "700" : "500",
        color: props.value() < 100 ? "var(--text)" : "var(--text-muted)",
        "min-width": "30px",
        "text-align": "right",
        "flex-shrink": "0",
      }}
    >
      {props.value()}%
    </span>
  </div>
);
