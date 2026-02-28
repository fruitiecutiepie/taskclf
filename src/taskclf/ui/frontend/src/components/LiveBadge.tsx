import { type Accessor, type Component, Show } from "solid-js";
import type { ConnectionStatus, LabelSuggestion, Prediction, StatusEvent, TrayState, WSStats } from "../lib/ws";
import { LABEL_COLORS, dotColor } from "./StatePanel";
import { host } from "../lib/host";

export const LiveBadge: Component<{
  status: Accessor<ConnectionStatus>;
  latestStatus: Accessor<StatusEvent | null>;
  latestPrediction: Accessor<Prediction | null>;
  latestTrayState: Accessor<TrayState | null>;
  activeSuggestion: Accessor<LabelSuggestion | null>;
  wsStats: Accessor<WSStats>;
  compact?: boolean;
  onShowPanel?: () => void;
  onHidePanel?: () => void;
  onShowLabel?: () => void;
  onHideLabel?: () => void;
}> = (props) => {
  const currentApp = () => props.latestStatus()?.current_app ?? null;

  const predictionLabel = () => {
    const pred = props.latestPrediction();
    return pred ? pred.mapped_label || pred.label : null;
  };

  const displayConfidence = () => {
    const pred = props.latestPrediction();
    return pred ? pred.confidence : null;
  };

  const predColor = () => {
    const l = predictionLabel();
    return l ? LABEL_COLORS[l] ?? "#555" : "#333";
  };

  return (
    <div
      style={{ display: "inline-block", width: "100%" }}
    >
      <div
        style={{
          display: "flex",
          "align-items": "center",
          gap: "8px",
          "justify-content": "center",
          width: "100%",
        }}
      >
        <span
          style={{
            padding: "3px 10px",
            "border-radius": "20px",
            "font-size": "0.75rem",
            "font-weight": "500",
            color: "#aaa",
            background: "#1a1a1a",
            border: "1px solid #333",
            "white-space": "nowrap",
            cursor: "default",
          }}
        >
          {currentApp() ?? "Unknown App"}
        </span>
        <span
          style={{
            padding: "3px 12px",
            "border-radius": "20px",
            "font-size": "0.85rem",
            "font-weight": "600",
            color: predictionLabel() ? "#fff" : "#888",
            background: predColor(),
            "white-space": "nowrap",
            cursor: "pointer",
          }}
          onMouseEnter={() => props.onShowLabel?.()}
          onMouseLeave={() => props.onHideLabel?.()}
        >
          {predictionLabel() ?? "Unknown Label"}
          <Show when={!props.compact && displayConfidence() !== null}>
            {" "}
            <span style={{ opacity: 0.8 }}>
              {Math.round(displayConfidence()! * 100)}%
            </span>
          </Show>
        </span>
        <span
          style={{
            width: "8px",
            height: "8px",
            "border-radius": "50%",
            background: dotColor(props.status()),
            "flex-shrink": "0",
            cursor: "pointer",
          }}
          title={props.status()}
          onMouseEnter={() => {
            host.invoke({ cmd: "showStatePanel" });
            props.onShowPanel?.();
          }}
          onMouseLeave={() => {
            host.invoke({ cmd: "hideStatePanel" });
            props.onHidePanel?.();
          }}
        />
      </div>
    </div>
  );
};
