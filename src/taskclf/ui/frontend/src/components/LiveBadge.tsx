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
}> = (props) => {
  const displayLabel = () => {
    const pred = props.latestPrediction();
    if (pred) return pred.mapped_label || pred.label;
    const st = props.latestStatus();
    if (st) return st.current_app;
    return null;
  };

  const displayConfidence = () => {
    const pred = props.latestPrediction();
    return pred ? pred.confidence : null;
  };

  const badgeColor = () => {
    const l = displayLabel();
    return l ? LABEL_COLORS[l] ?? "#555" : "#333";
  };

  return (
    <div
      style={{ display: "inline-block", width: "100%" }}
      onMouseEnter={() => host.invoke({ cmd: "showStatePanel" })}
      onMouseLeave={() => host.invoke({ cmd: "hideStatePanel" })}
    >
      <div
        style={{
          display: "flex",
          "align-items": "center",
          gap: "10px",
          "justify-content": "center",
          width: "100%",
        }}
      >
        <Show
          when={displayLabel()}
          fallback={
            <span
              style={{
                padding: "3px 16px",
                "border-radius": "20px",
                "font-size": "0.85rem",
                "font-weight": "600",
                color: "#888",
                background: "#1a1a1a",
                border: "1px solid #333",
                "white-space": "nowrap",
                cursor: "default",
              }}
            >
              No prediction
            </span>
          }
        >
          <span
            style={{
              padding: "3px 16px",
              "border-radius": "20px",
              "font-size": "0.85rem",
              "font-weight": "600",
              color: "#fff",
              background: badgeColor(),
              "white-space": "nowrap",
              cursor: "default",
            }}
          >
            {displayLabel()}
            <Show when={displayConfidence() !== null}>
              {" "}
              <span style={{ opacity: 0.8 }}>
                {Math.round(displayConfidence()! * 100)}%
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
    </div>
  );
};
