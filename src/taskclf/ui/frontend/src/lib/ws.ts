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
}

export type WSEvent = Prediction | LabelSuggestion | StatusEvent;

export type ConnectionStatus = "connecting" | "connected" | "disconnected";

export function useWebSocket() {
  const [latestPrediction, setLatestPrediction] =
    createSignal<Prediction | StatusEvent | null>(null);
  const [activeSuggestion, setActiveSuggestion] =
    createSignal<LabelSuggestion | null>(null);
  const [connectionStatus, setConnectionStatus] =
    createSignal<ConnectionStatus>("connecting");

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
      retryDelay = 1000;
    };

    ws.onmessage = (event) => {
      try {
        const data: WSEvent = JSON.parse(event.data);
        switch (data.type) {
          case "prediction":
          case "status":
            setLatestPrediction(data);
            break;
          case "suggest_label":
            setActiveSuggestion(data);
            break;
        }
      } catch {
        // ignore malformed messages
      }
    };

    ws.onclose = () => {
      setConnectionStatus("disconnected");
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
    latestPrediction,
    activeSuggestion,
    connectionStatus,
    dismissSuggestion,
  };
}
