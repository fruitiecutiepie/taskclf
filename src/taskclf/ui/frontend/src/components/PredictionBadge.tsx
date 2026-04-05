import type { Accessor, Component } from "solid-js";
import { LABEL_COLORS } from "../lib/labelColors";
import type {
  ConnectionStatus,
  LabelSuggestion,
  LiveStatusEvent,
  Prediction,
  StatusEvent,
  TrayState,
} from "../lib/ws";
import { ConnectionDot } from "./ConnectionDot";

export const PredictionBadge: Component<{
  status: Accessor<ConnectionStatus>;
  latest_status: Accessor<StatusEvent>;
  latest_prediction: Accessor<Prediction | null>;
  live_status: Accessor<LiveStatusEvent | null>;
  latest_tray_state: Accessor<TrayState>;
  active_suggestion: Accessor<LabelSuggestion | null>;
  label_pinned?: Accessor<boolean>;
  panel_pinned?: Accessor<boolean>;
  on_toggle_panel?: () => void;
  on_show_panel?: () => void;
  on_hide_panel?: () => void;
  on_toggle_label?: () => void;
  on_show_label?: () => void;
  on_hide_label?: () => void;
}> = (props) => {
  const prediction_label = () => {
    const pred = props.latest_prediction();
    return pred ? pred.mapped_label || pred.label : null;
  };

  const live_label = () => props.live_status()?.label ?? null;

  const display_label = () => prediction_label() ?? live_label();

  const no_model = () => !props.latest_tray_state().model_loaded;

  const badge_text = () =>
    display_label() ?? (no_model() ? "No Model" : "Unknown Label");

  const pred_color = () => {
    const l = display_label();
    return l ? (LABEL_COLORS[l] ?? "#555") : "#333";
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
      <button
        type="button"
        style={{
          padding: "2px 10px",
          "border-radius": "20px",
          "font-size": "0.75rem",
          "font-weight": "600",
          color: display_label() ? "#fff" : "#b0b0b0",
          background: pred_color(),
          "text-shadow": display_label() ? "0 1px 3px rgba(0,0,0,0.5)" : "none",
          "white-space": "nowrap",
          cursor: "pointer",
          outline: props.label_pinned?.() ? `2px solid ${pred_color()}aa` : "none",
          "outline-offset": "2px",
          transition: "outline 0.15s ease",
          border: "none",
          "font-family": "inherit",
          "line-height": "inherit",
        }}
        title={
          props.label_pinned?.()
            ? "Label grid pinned — click to unpin"
            : "Hover for label grid, click to pin"
        }
        onMouseEnter={() => props.on_show_label?.()}
        onMouseLeave={() => props.on_hide_label?.()}
        onClick={(e) => {
          e.stopPropagation();
          props.on_toggle_label?.();
        }}
      >
        {badge_text()}
      </button>
      <ConnectionDot
        status={props.status}
        panel_pinned={props.panel_pinned}
        on_toggle_panel={props.on_toggle_panel}
        on_show_panel={props.on_show_panel}
        on_hide_panel={props.on_hide_panel}
      />
    </div>
  );
};
