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
  latestStatus: Accessor<StatusEvent>;
  latestPrediction: Accessor<Prediction | null>;
  latestTrayState: Accessor<TrayState>;
  activeSuggestion: Accessor<LabelSuggestion | null>;
  wsStats: Accessor<WSStats>;
  trainState: Accessor<TrainState>;
}> = (props) => {
  const [tab, setTab] = createSignal<PanelTab>("system");

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
      <StatusPanelTab active={tab} onChange={setTab} />

      <Show when={tab() === "system"}>
        <StatusActivityMonitor status={props.latestStatus} />
        <StatusPrediction prediction={props.latestPrediction} />
        <StatusModel trayState={props.latestTrayState} />
        <StatusTransitions trayState={props.latestTrayState} />
        <StatusSuggestion suggestion={props.activeSuggestion} />
        <StatusActivityWatch status={props.latestStatus} />
        <StatusWebSocket status={props.status} wsStats={props.wsStats} />
        <StatusConfig trayState={props.latestTrayState} />
      </Show>

      <Show when={tab() === "history"}>
        <LabelHistory visible={() => true} latestPrediction={props.latestPrediction} />
      </Show>

      <Show when={tab() === "training"}>
        <TrainingPanel trainState={props.trainState} />
      </Show>
    </div>
  );
};
