import { type Accessor, type Component, createSignal, Show } from "solid-js";
import type {
  ConnectionStatus,
  LabelSuggestion,
  Prediction,
  StatusEvent,
  TrainState,
  TrayState,
  WSStats,
} from "../lib/ws";
import { LabelHistory } from "./LabelHistory";
import { StatusActivityMonitor } from "./status/StatusActivityMonitor";
import { StatusActivityWatch } from "./status/StatusActivityWatch";
import { StatusConfig } from "./status/StatusConfig";
import { StatusModel } from "./status/StatusModel";
import { type PanelTab, StatusPanelTab } from "./status/StatusPanelTab";
import { StatusPrediction } from "./status/StatusPrediction";
import { StatusSuggestion } from "./status/StatusSuggestion";
import { StatusTransitions } from "./status/StatusTransitions";
import { StatusWebSocket } from "./status/StatusWebSocket";
import { TrainingPanel } from "./TrainingPanel";

export const StatusPanel: Component<{
  status: Accessor<ConnectionStatus>;
  latest_status: Accessor<StatusEvent>;
  latest_prediction: Accessor<Prediction | null>;
  latest_tray_state: Accessor<TrayState>;
  active_suggestion: Accessor<LabelSuggestion | null>;
  ws_stats: Accessor<WSStats>;
  train_state: Accessor<TrainState>;
}> = (props) => {
  const [tab, set_tab] = createSignal<PanelTab>("system");

  return (
    <div
      style={{
        background: "var(--surface)",
        border: "1px solid #2a2a2a",
        "border-radius": "10px",
        padding: "6px 8px",
        "font-family": "'SF Mono', 'Fira Code', 'Cascadia Code', monospace",
        "font-size": "0.65rem",
        "min-width": "230px",
        color: "#e0e0e0",
        "box-shadow": "0 8px 32px rgba(0, 0, 0, 0.6)",
      }}
    >
      <StatusPanelTab active={tab} on_change={set_tab} />

      <Show when={tab() === "system"}>
        <StatusActivityMonitor status={props.latest_status} />
        <StatusPrediction prediction={props.latest_prediction} />
        <StatusModel tray_state={props.latest_tray_state} />
        <StatusTransitions tray_state={props.latest_tray_state} />
        <StatusSuggestion suggestion={props.active_suggestion} />
        <StatusActivityWatch status={props.latest_status} />
        <StatusWebSocket status={props.status} ws_stats={props.ws_stats} />
        <StatusConfig tray_state={props.latest_tray_state} />
      </Show>

      <Show when={tab() === "history"}>
        <LabelHistory
          visible={() => true}
          latest_prediction={props.latest_prediction}
        />
      </Show>

      <Show when={tab() === "training"}>
        <TrainingPanel train_state={props.train_state} />
      </Show>
    </div>
  );
};
