import { createSignal, onCleanup, onMount } from "solid-js";

export interface Prediction {
  type: "prediction";
  label: string;
  confidence: number;
  ts: string;
  mapped_label: string;
  current_app?: string;
}

export interface LabelSuggestion {
  type: "suggest_label";
  reason: string;
  old_label: string;
  suggested: string;
  confidence: number;
  block_start: string;
  block_end: string;
}

export interface StatusEvent {
  type: "status";
  state: "idle" | "collecting" | "predicting";
  current_app: string;
  current_app_since: string | null;
  candidate_app: string | null;
  candidate_duration_s: number;
  transition_threshold_s: number;
  poll_seconds: number;
  poll_count: number;
  last_poll_ts: string;
  uptime_s: number;
  aw_connected: boolean;
  aw_bucket_id: string | null;
  aw_host: string;
  last_event_count: number;
  last_app_counts: Record<string, number>;
}

export interface TransitionInfo {
  prev_app: string;
  new_app: string;
  block_start: string;
  block_end: string;
  fired_at: string;
}

export interface TrayState {
  type: "tray_state";
  model_loaded: boolean;
  model_dir: string | null;
  model_schema_hash: string | null;
  suggested_label: string | null;
  suggested_confidence: number | null;
  transition_count: number;
  last_transition: TransitionInfo | null;
  labels_saved_count: number;
  data_dir: string;
  ui_port: number;
  dev_mode: boolean;
}

export type WSEvent = Prediction | LabelSuggestion | StatusEvent | TrayState;

export type ConnectionStatus = "connecting" | "connected" | "disconnected";

export interface WSStats {
  messageCount: number;
  statusCount: number;
  predictionCount: number;
  trayStateCount: number;
  suggestionCount: number;
  lastMessageAt: string | null;
  reconnectCount: number;
  connectedSince: string | null;
}

export function useWebSocket() {
  const [latestStatus, setLatestStatus] = createSignal<StatusEvent | null>(
    null,
  );
  const [latestPrediction, setLatestPrediction] =
    createSignal<Prediction | null>(null);
  const [latestTrayState, setLatestTrayState] = createSignal<TrayState | null>(
    null,
  );
  const [activeSuggestion, setActiveSuggestion] =
    createSignal<LabelSuggestion | null>(null);
  const [connectionStatus, setConnectionStatus] =
    createSignal<ConnectionStatus>("connecting");
  const [wsStats, setWsStats] = createSignal<WSStats>({
    messageCount: 0,
    statusCount: 0,
    predictionCount: 0,
    trayStateCount: 0,
    suggestionCount: 0,
    lastMessageAt: null,
    reconnectCount: 0,
    connectedSince: null,
  });

  let ws: WebSocket | null = null;
  let reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  let retryDelay = 1000;

  function connect() {
    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    const url = `${proto}//${location.host}/ws/predictions`;

    setConnectionStatus("connecting");
    ws = new WebSocket(url);

    ws.onopen = () => {
      setConnectionStatus("connected");
      setWsStats((prev) => ({
        ...prev,
        connectedSince: new Date().toISOString(),
      }));
      retryDelay = 1000;
    };

    ws.onmessage = (event) => {
      try {
        const data: WSEvent = JSON.parse(event.data);
        const now = new Date().toISOString();
        switch (data.type) {
          case "status":
            setLatestStatus(data);
            setWsStats((prev) => ({
              ...prev,
              messageCount: prev.messageCount + 1,
              statusCount: prev.statusCount + 1,
              lastMessageAt: now,
            }));
            break;
          case "prediction":
            setLatestPrediction(data);
            setWsStats((prev) => ({
              ...prev,
              messageCount: prev.messageCount + 1,
              predictionCount: prev.predictionCount + 1,
              lastMessageAt: now,
            }));
            break;
          case "tray_state":
            setLatestTrayState(data);
            setWsStats((prev) => ({
              ...prev,
              messageCount: prev.messageCount + 1,
              trayStateCount: prev.trayStateCount + 1,
              lastMessageAt: now,
            }));
            break;
          case "suggest_label":
            setActiveSuggestion(data);
            setWsStats((prev) => ({
              ...prev,
              messageCount: prev.messageCount + 1,
              suggestionCount: prev.suggestionCount + 1,
              lastMessageAt: now,
            }));
            break;
        }
      } catch {
        // ignore malformed messages
      }
    };

    ws.onclose = () => {
      setConnectionStatus("disconnected");
      setWsStats((prev) => ({ ...prev, connectedSince: null }));
      scheduleReconnect();
    };

    ws.onerror = () => {
      ws?.close();
    };
  }

  function scheduleReconnect() {
    if (reconnectTimer) return;
    reconnectTimer = setTimeout(() => {
      reconnectTimer = null;
      retryDelay = Math.min(retryDelay * 2, 30000);
      setWsStats((prev) => ({
        ...prev,
        reconnectCount: prev.reconnectCount + 1,
      }));
      connect();
    }, retryDelay);
  }

  function dismissSuggestion() {
    setActiveSuggestion(null);
  }

  onMount(() => connect());

  onCleanup(() => {
    if (reconnectTimer) clearTimeout(reconnectTimer);
    ws?.close();
  });

  return {
    latestStatus,
    latestPrediction,
    latestTrayState,
    activeSuggestion,
    connectionStatus,
    wsStats,
    dismissSuggestion,
  };
}
