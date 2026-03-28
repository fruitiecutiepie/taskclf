import { onCleanup, onMount } from "solid-js";
import { createStore, produce, reconcile } from "solid-js/store";

export type Prediction = {
  type: "prediction";
  label: string;
  confidence: number;
  ts: string;
  mapped_label: string;
  current_app?: string;
  provenance?: "manual" | "model";
};

export type LabelSuggestion = {
  type: "suggest_label";
  reason: string;
  old_label: string;
  suggested: string;
  confidence: number;
  block_start: string;
  block_end: string;
};

export type StatusEvent = {
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
};
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

export type TransitionInfo = {
  prev_app: string;
  new_app: string;
  block_start: string;
  block_end: string;
  fired_at: string;
};

export type TrayState = {
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
};
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

export type ShowLabelGridEvent = {
  type: "label_grid_show";
};

export type PromptLabelEvent = {
  type: "prompt_label";
  prev_app: string;
  new_app: string;
  block_start: string;
  block_end: string;
  duration_min: number;
  suggested_label: string | null;
  suggestion_text: string | null;
};

export type SuggestionClearedEvent = {
  type: "suggestion_cleared";
  reason: string;
};

export type NoModelTransitionEvent = {
  type: "no_model_transition";
  current_app: string;
  ts: string;
  block_start: string;
  block_end: string;
};

export type LabelCreatedEvent = {
  type: "label_created";
  label: string;
  confidence: number;
  ts: string;
  start_ts: string;
  extend_forward: boolean;
};

export type LiveStatusEvent = {
  type: "live_status";
  label: string;
  text: string;
  ts: string;
};

export type TrainProgressEvent = {
  type: "train_progress";
  job_id: string;
  step: string;
  progress_pct: number | null;
  message: string | null;
};

export type TrainCompleteEvent = {
  type: "train_complete";
  job_id: string;
  metrics: { macro_f1?: number; weighted_f1?: number } | null;
  model_dir: string | null;
};

export type TrainFailedEvent = {
  type: "train_failed";
  job_id: string;
  error: string;
};

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
  | LiveStatusEvent
  | TrainProgressEvent
  | TrainCompleteEvent
  | TrainFailedEvent;

export type ConnectionStatus = "connecting" | "connected" | "disconnected";

export type TrainState = {
  job_id: string | null;
  status: "idle" | "running" | "complete" | "failed";
  step: string | null;
  progress_pct: number | null;
  message: string | null;
  error: string | null;
  metrics: { macro_f1?: number; weighted_f1?: number } | null;
  model_dir: string | null;
};

export type WSStats = {
  message_count: number;
  status_count: number;
  prediction_count: number;
  tray_state_count: number;
  suggestion_count: number;
  last_message_at: string | null;
  reconnect_count: number;
  connected_since: string | null;
};

export type WebSocketStore = {
  latest_status: StatusEvent;
  latest_prediction: Prediction | null;
  latest_tray_state: TrayState;
  active_suggestion: LabelSuggestion | null;
  latest_prompt: PromptLabelEvent | null;
  live_status: LiveStatusEvent | null;
  label_grid_requested: number;
  train_state: TrainState;
  connection_status: ConnectionStatus;
  ws_stats: WSStats;
};

export function ws_store_new() {
  const [store, setStore] = createStore<WebSocketStore>({
    latest_status: StatusEventDefault,
    latest_prediction: null,
    latest_tray_state: TrayStateDefault,
    active_suggestion: null,
    latest_prompt: null,
    live_status: null,
    label_grid_requested: 0,
    train_state: {
      job_id: null,
      status: "idle",
      step: null,
      progress_pct: null,
      message: null,
      error: null,
      metrics: null,
      model_dir: null,
    },
    connection_status: "connecting",
    ws_stats: {
      message_count: 0,
      status_count: 0,
      prediction_count: 0,
      tray_state_count: 0,
      suggestion_count: 0,
      last_message_at: null,
      reconnect_count: 0,
      connected_since: null,
    },
  });

  const SUGGESTION_TTL_MS = 10 * 60 * 1000;

  let ws: WebSocket | null = null;
  let reconnect_timer: ReturnType<typeof setTimeout> | null = null;
  let suggestion_timer: ReturnType<typeof setTimeout> | null = null;
  let retry_delay = 1000;

  function suggestion_timer_clear() {
    if (suggestion_timer) {
      clearTimeout(suggestion_timer);
      suggestion_timer = null;
    }
  }

  function suggestion_timer_start() {
    suggestion_timer_clear();
    suggestion_timer = setTimeout(() => {
      suggestion_timer = null;
      setStore("active_suggestion", null);
    }, SUGGESTION_TTL_MS);
  }

  function ws_stats_bump(now: string, extra?: (s: WSStats) => void) {
    setStore(
      "ws_stats",
      produce((s) => {
        s.message_count++;
        s.last_message_at = now;
        extra?.(s);
      }),
    );
  }

  async function ws_snapshot_hydrate() {
    try {
      const resp = await fetch("/api/ws/snapshot");
      if (!resp.ok) {
        return;
      }
      const snap: Record<string, WSEvent> = await resp.json();
      if (snap.status) {
        setStore("latest_status", reconcile(snap.status as StatusEvent));
      }
      if (snap.prediction) {
        setStore("latest_prediction", reconcile(snap.prediction as Prediction | null));
      }
      if (snap.tray_state) {
        setStore("latest_tray_state", snap.tray_state as TrayState);
      }
    } catch {
      // hydration is best-effort; the next push will update us
    }
  }

  function ws_connection_open() {
    if (
      ws
      && (ws.readyState === WebSocket.CONNECTING || ws.readyState === WebSocket.OPEN)
    ) {
      return;
    }

    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    const url = `${proto}//${location.host}/ws/predictions`;

    setStore("connection_status", "connecting");
    ws = new WebSocket(url);

    ws.onopen = () => {
      setStore("connection_status", "connected");
      setStore("ws_stats", "connected_since", new Date().toISOString());
      retry_delay = 1000;
      ws_snapshot_hydrate();
    };

    ws.onmessage = (event) => {
      try {
        const data: WSEvent = JSON.parse(event.data);
        const now = new Date().toISOString();
        switch (data.type) {
          case "status":
            setStore("latest_status", reconcile(data));
            ws_stats_bump(now, (s) => {
              s.status_count++;
            });
            break;
          case "prediction":
            setStore("latest_prediction", reconcile(data as Prediction | null));
            ws_stats_bump(now, (s) => {
              s.prediction_count++;
            });
            break;
          case "tray_state":
            setStore("latest_tray_state", data);
            ws_stats_bump(now, (s) => {
              s.tray_state_count++;
            });
            break;
          case "suggest_label":
            setStore("active_suggestion", data);
            suggestion_timer_start();
            ws_stats_bump(now, (s) => {
              s.suggestion_count++;
            });
            break;
          case "suggestion_cleared":
            setStore("active_suggestion", null);
            suggestion_timer_clear();
            ws_stats_bump(now);
            break;
          case "label_grid_show":
            setStore("label_grid_requested", (n) => n + 1);
            ws_stats_bump(now);
            break;
          case "prompt_label":
            setStore("latest_prompt", data);
            ws_stats_bump(now);
            break;
          case "label_created":
            setStore(
              "latest_prediction",
              reconcile({
                type: "prediction",
                label: data.label,
                confidence: data.confidence,
                ts: data.ts,
                mapped_label: data.label,
                provenance: "manual",
              } as Prediction | null),
            );
            ws_stats_bump(now, (s) => {
              s.prediction_count++;
            });
            break;
          case "no_model_transition":
            ws_stats_bump(now);
            break;
          case "live_status":
            setStore("live_status", data as LiveStatusEvent);
            ws_stats_bump(now);
            break;
          case "train_progress":
            setStore(
              "train_state",
              produce((t) => {
                t.job_id = data.job_id;
                t.status = "running";
                t.step = data.step;
                t.progress_pct = data.progress_pct;
                t.message = data.message;
              }),
            );
            ws_stats_bump(now);
            break;
          case "train_complete":
            setStore(
              "train_state",
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
            ws_stats_bump(now);
            break;
          case "train_failed":
            setStore(
              "train_state",
              produce((t) => {
                t.job_id = data.job_id;
                t.status = "failed";
                t.step = null;
                t.progress_pct = null;
                t.message = null;
                t.error = data.error;
              }),
            );
            ws_stats_bump(now);
            break;
        }
      } catch {
        // ignore malformed messages
      }
    };

    ws.onclose = () => {
      setStore("connection_status", "disconnected");
      setStore("ws_stats", "connected_since", null);
      reconnect_schedule();
    };

    ws.onerror = () => {
      ws?.close();
    };
  }

  function reconnect_schedule() {
    if (reconnect_timer) {
      return;
    }
    if (typeof navigator !== "undefined" && !navigator.onLine) {
      // Offline — don't burn retries; the 'online' listener will trigger connect()
      return;
    }
    const jitter = retry_delay * (0.5 + Math.random() * 0.5);
    reconnect_timer = setTimeout(() => {
      reconnect_timer = null;
      retry_delay = Math.min(retry_delay * 2, 30_000);
      setStore("ws_stats", "reconnect_count", (c) => c + 1);
      ws_connection_open();
    }, jitter);
  }

  function online_event_sync() {
    retry_delay = 1000;
    if (reconnect_timer) {
      clearTimeout(reconnect_timer);
      reconnect_timer = null;
    }
    ws_connection_open();
  }

  function visibility_event_sync() {
    if (
      document.visibilityState === "visible"
      && store.connection_status === "disconnected"
    ) {
      retry_delay = 1000;
      if (reconnect_timer) {
        clearTimeout(reconnect_timer);
        reconnect_timer = null;
      }
      ws_connection_open();
    }
  }

  function suggestion_dismiss() {
    setStore("active_suggestion", null);
    suggestion_timer_clear();
  }

  onMount(() => {
    ws_connection_open();
    window.addEventListener("online", online_event_sync);
    document.addEventListener("visibilitychange", visibility_event_sync);
  });

  onCleanup(() => {
    window.removeEventListener("online", online_event_sync);
    document.removeEventListener("visibilitychange", visibility_event_sync);
    if (reconnect_timer) {
      clearTimeout(reconnect_timer);
    }
    suggestion_timer_clear();
    ws?.close();
  });

  return {
    latest_status: () => store.latest_status,
    latest_prediction: () => store.latest_prediction,
    latest_tray_state: () => store.latest_tray_state,
    active_suggestion: () => store.active_suggestion,
    latest_prompt: () => store.latest_prompt,
    label_grid_requested: () => store.label_grid_requested,
    connection_status: () => store.connection_status,
    ws_stats: () => store.ws_stats,
    train_state: () => store.train_state,
    suggestion_dismiss,
  };
}
