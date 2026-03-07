import { type Accessor, type Component } from "solid-js";
import type { ConnectionStatus, LabelSuggestion, Prediction, StatusEvent, TrayState } from "../lib/ws";
import { LABEL_COLORS } from "./StatePanel";
import { LiveBadgeConnectionStatus } from "./LiveBadgeConnectionStatus";

export const LiveBadge: Component<{
  status: Accessor<ConnectionStatus>;
  latestStatus: Accessor<StatusEvent | null>;
  latestPrediction: Accessor<Prediction | null>;
  latestTrayState: Accessor<TrayState | null>;
  activeSuggestion: Accessor<LabelSuggestion | null>;
  labelPinned?: Accessor<boolean>;
  panelPinned?: Accessor<boolean>;
  onTogglePanel?: () => void;
  onShowPanel?: () => void;
  onHidePanel?: () => void;
  onToggleLabel?: () => void;
  onShowLabel?: () => void;
  onHideLabel?: () => void;
}> = (props) => {
  const predictionLabel = () => {
    const pred = props.latestPrediction();
    return pred ? pred.mapped_label || pred.label : null;
  };

  const noModel = () => {
    const tray = props.latestTrayState();
    return tray !== null && !tray.model_loaded;
  };

  const badgeText = () => predictionLabel() ?? (noModel() ? "No Model" : "Unknown Label");

  const predColor = () => {
    const l = predictionLabel();
    return l ? LABEL_COLORS[l] ?? "#555" : "#333";
  };

  return (
    <div
      style={{
        display: "flex",
        "align-items": "center",
        "justify-content": "center",
        width: "100%",
        gap: "4px",
      }}
    >
      <span
        style={{
          padding: "2px 10px",
          "border-radius": "20px",
          "font-size": "0.75rem",
          "font-weight": "600",
          color: predictionLabel() ? "#fff" : "#b0b0b0",
          background: predColor(),
          "text-shadow": predictionLabel() ? "0 1px 3px rgba(0,0,0,0.5)" : "none",
          "white-space": "nowrap",
          cursor: "pointer",
          outline: props.labelPinned?.()
            ? `2px solid ${predColor()}aa`
            : "none",
          "outline-offset": "2px",
          transition: "outline 0.15s ease",
        }}
        title={props.labelPinned?.() ? "Label grid pinned — click to unpin" : "Hover for label grid, click to pin"}
        onMouseEnter={() => props.onShowLabel?.()}
        onMouseLeave={() => props.onHideLabel?.()}
        onClick={(e) => {
          e.stopPropagation();
          props.onToggleLabel?.();
        }}
      >
        {badgeText()}
      </span>
      <LiveBadgeConnectionStatus
        status={props.status}
        panelPinned={props.panelPinned}
        onTogglePanel={props.onTogglePanel}
        onShowPanel={props.onShowPanel}
        onHidePanel={props.onHidePanel}
      />
    </div>
  );
};
