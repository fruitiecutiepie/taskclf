import { type Accessor, type Component, Show } from "solid-js";
import type { ConnectionStatus, LabelSuggestion, Prediction, StatusEvent, TrayState, WSStats } from "../lib/ws";
import { LABEL_COLORS } from "./StatePanel";
import { LiveBadgeConnectionStatus } from "./LiveBadgeConnectionStatus";

export const LiveBadge: Component<{
  status: Accessor<ConnectionStatus>;
  latestStatus: Accessor<StatusEvent | null>;
  latestPrediction: Accessor<Prediction | null>;
  latestTrayState: Accessor<TrayState | null>;
  activeSuggestion: Accessor<LabelSuggestion | null>;
  wsStats: Accessor<WSStats>;
  compact?: boolean;
  historyOpen?: Accessor<boolean>;
  onTogglePanel?: () => void;
  onToggleHistory?: () => void;
  onShowLabel?: () => void;
  onHideLabel?: () => void;
}> = (props) => {
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
      style={{
        display: "flex",
        "align-items": "center",
        "justify-content": "space-between",
        width: "100%",
      }}
    >
      {/* Invisible spacer matching the button width to keep center truly centered */}
      <div style={{ width: "32px", "flex-shrink": "0" }} />

      <div
        style={{
          display: "flex",
          "align-items": "center",
          gap: "8px",
          "justify-content": "center",
        }}
      >
        <span
          style={{
            padding: "3px 12px",
            "border-radius": "20px",
            "font-size": "0.85rem",
            "font-weight": "600",
            color: predictionLabel() ? "#fff" : "#b0b0b0",
            background: predColor(),
            "text-shadow": predictionLabel() ? "0 1px 3px rgba(0,0,0,0.5)" : "none",
            "white-space": "nowrap",
            cursor: "pointer",
          }}
          onMouseEnter={() => props.onShowLabel?.()}
          onMouseLeave={() => props.onHideLabel?.()}
        >
          {predictionLabel() ?? "Unknown Label"}
          <Show when={!props.compact && displayConfidence() !== null}>
            {" "}
            <span style={{ opacity: 0.9 }}>
              {Math.round(displayConfidence()! * 100)}%
            </span>
          </Show>
        </span>
        <LiveBadgeConnectionStatus
          status={props.status}
          onTogglePanel={props.onTogglePanel}
        />
      </div>

      <Show when={props.onToggleHistory}>
        <button
          onClick={(e) => {
            e.stopPropagation();
            props.onToggleHistory?.();
          }}
          style={{
            background: "none",
            border: "none",
            color: "var(--text)",
            cursor: "pointer",
            "font-size": "0.9rem",
            padding: "4px 8px",
            "line-height": "1",
            transform: props.historyOpen?.() ? "rotate(180deg)" : "none",
            transition: "transform 0.15s ease",
          }}
          title={props.historyOpen?.() ? "Hide history" : "Show history"}
        >
          &#9660;
        </button>
      </Show>
    </div>
  );
};
