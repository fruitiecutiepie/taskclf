import { type Accessor, type Component, Show } from "solid-js";
import type { ConnectionStatus, Prediction, StatusEvent } from "../lib/ws";

const LABEL_COLORS: Record<string, string> = {
  Build: "#6366f1",
  Debug: "#f59e0b",
  Review: "#8b5cf6",
  Write: "#3b82f6",
  ReadResearch: "#14b8a6",
  Communicate: "#f97316",
  Meet: "#ec4899",
  BreakIdle: "#6b7280",
  "Mixed/Unknown": "#6b7280",
  unknown: "#6b7280",
};

function dotColor(status: ConnectionStatus): string {
  switch (status) {
    case "connected":
      return "var(--success)";
    case "connecting":
      return "var(--warning)";
    case "disconnected":
      return "var(--danger)";
  }
}

export const LiveBadge: Component<{
  prediction: Accessor<Prediction | StatusEvent | null>;
  status: Accessor<ConnectionStatus>;
}> = (props) => {
  const label = () => {
    const p = props.prediction();
    if (!p) return null;
    if (p.type === "status") return p.current_app;
    return p.mapped_label || p.label;
  };

  const confidence = () => {
    const p = props.prediction();
    if (!p || p.type === "status") return null;
    return p.confidence;
  };

  const badgeColor = () => {
    const l = label();
    return l ? LABEL_COLORS[l] ?? "var(--accent)" : "var(--border)";
  };

  return (
    <div
      style={{
        display: "flex",
        "align-items": "center",
        gap: "10px",
      }}
    >
      <Show when={label()}>
        <span
          style={{
            padding: "4px 12px",
            "border-radius": "20px",
            "font-size": "0.8rem",
            "font-weight": "600",
            color: "#fff",
            background: badgeColor(),
          }}
        >
          {label()}
          <Show when={confidence() !== null}>
            {" "}
            <span style={{ opacity: 0.8 }}>
              {Math.round(confidence()! * 100)}%
            </span>
          </Show>
        </span>
      </Show>
      <span
        style={{
          width: "8px",
          height: "8px",
          "border-radius": "50%",
          background: dotColor(props.status()),
          "flex-shrink": "0",
        }}
        title={props.status()}
      />
    </div>
  );
};
