import { onCleanup, onMount } from "solid-js";
import { createStore, produce, reconcile } from "solid-js/store";

export interface Prediction {
  type: "prediction";
  label: string;
  confidence: number;
  ts: string;
  mapped_label: string;
  current_app?: string;
  provenance?: "manual" | "model";
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
  state: "idle" | "collecting" | "predicting" | "paused";
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
const StatusEventDefault: StatusEvent = {
  type: "status",
  state: "idle",
  current_app: "unknown",
  current_app_since: null,
  candidate_app: null,
  candidate_duration_s: 0,
  transition_threshold_s: 0,
  poll_seconds: 0,
  poll_count: 0,
  last_poll_ts: new Date(0).toISOString(),
  uptime_s: 0,
  aw_connected: false,
  aw_bucket_id: null,
  aw_host: "http://localhost:5600",
  last_event_count: 0,
  last_app_counts: {},
} as const;

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
  paused: boolean;
}
const TrayStateDefault: TrayState = {
  type: "tray_state",
  model_loaded: false,
  model_dir: null,
  model_schema_hash: null,
  suggested_label: null,
  suggested_confidence: null,
  transition_count: 0,
  last_transition: null,
  labels_saved_count: 0,
  data_dir: "~/.taskclf",
  ui_port: 0,
  dev_mode: false,
  paused: false,
} as const;

export interface ShowLabelGridEvent {
  type: "label_grid_show";
}

export interface PromptLabelEvent {
  type: "prompt_label";
  prev_app: string;
  new_app: string;
  block_start: string;
  block_end: string;
  duration_min: number;
  suggested_label: string | null;
  suggested_confidence: number | null;
}

export interface SuggestionClearedEvent {
  type: "suggestion_cleared";
  reason: string;
}

export interface NoModelTransitionEvent {
  type: "no_model_transition";
  current_app: string;
  ts: string;
  block_start: string;
  block_end: string;
}

export interface LabelCreatedEvent {
  type: "label_created";
  label: string;
  confidence: number;
  ts: string;
  start_ts: string;
  extend_forward: boolean;
}

export interface TrainProgressEvent {
  type: "train_progress";
  job_id: string;
  step: string;
  progress_pct: number | null;
  message: string | null;
}

export interface TrainCompleteEvent {
  type: "train_complete";
  job_id: string;
  metrics: { macro_f1?: number; weighted_f1?: number } | null;
  model_dir: string | null;
}

export interface TrainFailedEvent {
  type: "train_failed";
  job_id: string;
  error: string;
}

export type WSEvent =
  | Prediction
  | LabelSuggestion
  | StatusEvent
  | TrayState
  | ShowLabelGridEvent
  | PromptLabelEvent
  | SuggestionClearedEvent
  | NoModelTransitionEvent
  | LabelCreatedEvent
  | TrainProgressEvent
  | TrainCompleteEvent
  | TrainFailedEvent;

export type ConnectionStatus = "connecting" | "connected" | "disconnected";

export interface TrainState {
  job_id: string | null;
  status: "idle" | "running" | "complete" | "failed";
  step: string | null;
  progress_pct: number | null;
  message: string | null;
  error: string | null;
  metrics: { macro_f1?: number; weighted_f1?: number } | null;
  model_dir: string | null;
}

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

export interface WebSocketStore {
  latestStatus: StatusEvent;
  latestPrediction: Prediction | null;
  latestTrayState: TrayState;
  activeSuggestion: LabelSuggestion | null;
  latestPrompt: PromptLabelEvent | null;
  labelGridRequested: number;
  trainState: TrainState;
  connectionStatus: ConnectionStatus;
  wsStats: WSStats;
}

export function useWebSocket() {
  const [store, setStore] = createStore<WebSocketStore>({
    latestStatus: StatusEventDefault,
    latestPrediction: null,
    latestTrayState: TrayStateDefault,
    activeSuggestion: null,
    latestPrompt: null,
    labelGridRequested: 0,
    trainState: {
      job_id: null,
      status: "idle",
      step: null,
      progress_pct: null,
      message: null,
      error: null,
      metrics: null,
      model_dir: null,
    },
    connectionStatus: "connecting",
    wsStats: {
      messageCount: 0,
      statusCount: 0,
      predictionCount: 0,
      trayStateCount: 0,
      suggestionCount: 0,
      lastMessageAt: null,
      reconnectCount: 0,
      connectedSince: null,
    },
  });

  const SUGGESTION_TTL_MS = 10 * 60 * 1000;

  let ws: WebSocket | null = null;
  let reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  let suggestionTimer: ReturnType<typeof setTimeout> | null = null;
  let retryDelay = 1000;

  function clearSuggestionTimer() {
    if (suggestionTimer) {
      clearTimeout(suggestionTimer);
      suggestionTimer = null;
    }
  }

  function startSuggestionTimer() {
    clearSuggestionTimer();
    suggestionTimer = setTimeout(() => {
      suggestionTimer = null;
      setStore("activeSuggestion", null);
    }, SUGGESTION_TTL_MS);
  }

  function bumpStats(now: string, extra?: (s: WSStats) => void) {
    setStore(
      "wsStats",
      produce((s) => {
        s.messageCount++;
        s.lastMessageAt = now;
        extra?.(s);
      }),
    );
  }

  async function hydrateFromSnapshot() {
    try {
      const resp = await fetch("/api/ws/snapshot");
      if (!resp.ok) {
        return;
      }
      const snap: Record<string, WSEvent> = await resp.json();
      if (snap.status) {
        setStore("latestStatus", reconcile(snap.status as StatusEvent));
      }
      if (snap.prediction) {
        setStore("latestPrediction", reconcile(snap.prediction as Prediction | null));
      }
      if (snap.tray_state) {
        setStore("latestTrayState", snap.tray_state as TrayState);
      }
    } catch {
      // hydration is best-effort; the next push will update us
    }
  }

  function connect() {
    if (
      ws
      && (ws.readyState === WebSocket.CONNECTING || ws.readyState === WebSocket.OPEN)
    ) {
      return;
    }

    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    const url = `${proto}//${location.host}/ws/predictions`;

    setStore("connectionStatus", "connecting");
    ws = new WebSocket(url);

    ws.onopen = () => {
      setStore("connectionStatus", "connected");
      setStore("wsStats", "connectedSince", new Date().toISOString());
      retryDelay = 1000;
      hydrateFromSnapshot();
    };

    ws.onmessage = (event) => {
      try {
        const data: WSEvent = JSON.parse(event.data);
        const now = new Date().toISOString();
        switch (data.type) {
          case "status":
            setStore("latestStatus", reconcile(data));
            bumpStats(now, (s) => {
              s.statusCount++;
            });
            break;
          case "prediction":
            setStore("latestPrediction", reconcile(data as Prediction | null));
            bumpStats(now, (s) => {
              s.predictionCount++;
            });
            break;
          case "tray_state":
            setStore("latestTrayState", data);
            bumpStats(now, (s) => {
              s.trayStateCount++;
            });
            break;
          case "suggest_label":
            setStore("activeSuggestion", data);
            startSuggestionTimer();
            bumpStats(now, (s) => {
              s.suggestionCount++;
            });
            break;
          case "suggestion_cleared":
            setStore("activeSuggestion", null);
            clearSuggestionTimer();
            bumpStats(now);
            break;
          case "label_grid_show":
            setStore("labelGridRequested", (n) => n + 1);
            bumpStats(now);
            break;
          case "prompt_label":
            setStore("latestPrompt", data);
            bumpStats(now);
            break;
          case "label_created":
            setStore(
              "latestPrediction",
              reconcile({
                type: "prediction",
                label: data.label,
                confidence: data.confidence,
                ts: data.ts,
                mapped_label: data.label,
                provenance: "manual",
              } as Prediction | null),
            );
            bumpStats(now, (s) => {
              s.predictionCount++;
            });
            break;
          case "no_model_transition":
            bumpStats(now);
            break;
          case "train_progress":
            setStore(
              "trainState",
              produce((t) => {
                t.job_id = data.job_id;
                t.status = "running";
                t.step = data.step;
                t.progress_pct = data.progress_pct;
                t.message = data.message;
              }),
            );
            bumpStats(now);
            break;
          case "train_complete":
            setStore(
              "trainState",
              produce((t) => {
                t.job_id = data.job_id;
                t.status = "complete";
                t.step = "done";
                t.progress_pct = 100;
                t.message = null;
                t.error = null;
                t.metrics = data.metrics;
                t.model_dir = data.model_dir;
              }),
            );
            bumpStats(now);
            break;
          case "train_failed":
            setStore(
              "trainState",
              produce((t) => {
                t.job_id = data.job_id;
                t.status = "failed";
                t.step = null;
                t.progress_pct = null;
                t.message = null;
                t.error = data.error;
              }),
            );
            bumpStats(now);
            break;
        }
      } catch {
        // ignore malformed messages
      }
    };

    ws.onclose = () => {
      setStore("connectionStatus", "disconnected");
      setStore("wsStats", "connectedSince", null);
      scheduleReconnect();
    };

    ws.onerror = () => {
      ws?.close();
    };
  }

  function scheduleReconnect() {
    if (reconnectTimer) {
      return;
    }
    if (typeof navigator !== "undefined" && !navigator.onLine) {
      // Offline — don't burn retries; the 'online' listener will trigger connect()
      return;
    }
    const jitter = retryDelay * (0.5 + Math.random() * 0.5);
    reconnectTimer = setTimeout(() => {
      reconnectTimer = null;
      retryDelay = Math.min(retryDelay * 2, 30_000);
      setStore("wsStats", "reconnectCount", (c) => c + 1);
      connect();
    }, jitter);
  }

  function onOnline() {
    retryDelay = 1000;
    if (reconnectTimer) {
      clearTimeout(reconnectTimer);
      reconnectTimer = null;
    }
    connect();
  }

  function onVisibilityChange() {
    if (
      document.visibilityState === "visible"
      && store.connectionStatus === "disconnected"
    ) {
      retryDelay = 1000;
      if (reconnectTimer) {
        clearTimeout(reconnectTimer);
        reconnectTimer = null;
      }
      connect();
    }
  }

  function dismissSuggestion() {
    setStore("activeSuggestion", null);
    clearSuggestionTimer();
  }

  onMount(() => {
    connect();
    window.addEventListener("online", onOnline);
    document.addEventListener("visibilitychange", onVisibilityChange);
  });

  onCleanup(() => {
    window.removeEventListener("online", onOnline);
    document.removeEventListener("visibilitychange", onVisibilityChange);
    if (reconnectTimer) {
      clearTimeout(reconnectTimer);
    }
    clearSuggestionTimer();
    ws?.close();
  });

  return {
    latestStatus: () => store.latestStatus,
    latestPrediction: () => store.latestPrediction,
    latestTrayState: () => store.latestTrayState,
    activeSuggestion: () => store.activeSuggestion,
    latestPrompt: () => store.latestPrompt,
    labelGridRequested: () => store.labelGridRequested,
    connectionStatus: () => store.connectionStatus,
    wsStats: () => store.wsStats,
    trainState: () => store.trainState,
    dismissSuggestion,
  };
}
