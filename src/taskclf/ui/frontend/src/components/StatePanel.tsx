import {
  type Accessor,
  type Component,
  For,
  Show,
  createMemo,
} from "solid-js";
import type {
  ConnectionStatus,
  LabelSuggestion,
  Prediction,
  StatusEvent,
  TrayState,
  WSStats,
} from "../lib/ws";

export const LABEL_COLORS: Record<string, string> = {
  Build: "#6366f1",
  Debug: "#f59e0b",
  Review: "#8b5cf6",
  Write: "#3b82f6",
  ReadResearch: "#14b8a6",
  Communicate: "#f97316",
  Meet: "#ec4899",
  BreakIdle: "#6b7280",
  "Mixed/Unknown": "#6b7280",
  unknown: "#6b7280",
};

export function dotColor(status: ConnectionStatus): string {
  switch (status) {
    case "connected":
      return "#22c55e";
    case "connecting":
      return "#eab308";
    case "disconnected":
      return "#ef4444";
  }
}

function formatDuration(seconds: number): string {
  if (seconds < 60) return `${seconds}s`;
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  if (m < 60) return s > 0 ? `${m}m ${s}s` : `${m}m`;
  const h = Math.floor(m / 60);
  const rm = m % 60;
  return rm > 0 ? `${h}h ${rm}m` : `${h}h`;
}

function formatTime(iso: string | null | undefined): string {
  if (!iso) return "—";
  try {
    const d = new Date(iso.includes("Z") || iso.includes("+") ? iso : iso + "Z");
    return d.toLocaleTimeString([], {
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    });
  } catch {
    return iso;
  }
}

function truncPath(p: string | null | undefined, maxLen = 30): string {
  if (!p) return "—";
  if (p.length <= maxLen) return p;
  return "…" + p.slice(-(maxLen - 1));
}

// ---------------------------------------------------------------------------
// Row & Section primitives
// ---------------------------------------------------------------------------

const Row: Component<{
  label: string;
  value: string;
  color?: string;
  dim?: boolean;
  mono?: boolean;
}> = (props) => (
  <div
    style={{
      display: "flex",
      "justify-content": "space-between",
      "align-items": "baseline",
      padding: "1px 0",
      gap: "8px",
    }}
  >
    <span
      style={{
        color: "#8a8a8a",
        "font-size": "0.65rem",
        "flex-shrink": "0",
        "user-select": "none",
      }}
    >
      {props.label}
    </span>
    <span
      title={props.value}
      style={{
        "font-size": "0.65rem",
        "font-weight": props.dim ? "400" : "600",
        "font-family": props.mono
          ? "'SF Mono', 'Fira Code', monospace"
          : "inherit",
        color: props.color ?? (props.dim ? "#8a8a8a" : "#d0d0d0"),
        "text-align": "right",
        overflow: "hidden",
        "text-overflow": "ellipsis",
        "white-space": "nowrap",
        "min-width": "0",
      }}
    >
      {props.value}
    </span>
  </div>
);

const Section: Component<{ title: string; children: any }> = (props) => (
  <div style={{ "margin-bottom": "5px" }}>
    <div
      style={{
        "font-size": "0.6rem",
        "font-weight": "700",
        "text-transform": "uppercase",
        "letter-spacing": "0.06em",
        color: "#7a7a7a",
        "margin-bottom": "1px",
        "border-bottom": "1px solid #333",
        "padding-bottom": "1px",
      }}
    >
      {props.title}
    </div>
    {props.children}
  </div>
);

const ProgressBar: Component<{ pct: number; color?: string }> = (props) => (
  <div
    style={{
      height: "3px",
      background: "#2a2a2a",
      "border-radius": "2px",
      overflow: "hidden",
      "margin-top": "2px",
    }}
  >
    <div
      style={{
        height: "100%",
        width: `${Math.min(100, props.pct)}%`,
        background: props.color ?? "#eab308",
        "border-radius": "2px",
        transition: "width 0.3s ease",
      }}
    />
  </div>
);

// ---------------------------------------------------------------------------
// StatePanel
// ---------------------------------------------------------------------------

export const StatePanel: Component<{
  status: Accessor<ConnectionStatus>;
  latestStatus: Accessor<StatusEvent | null>;
  latestPrediction: Accessor<Prediction | null>;
  latestTrayState: Accessor<TrayState | null>;
  activeSuggestion: Accessor<LabelSuggestion | null>;
  wsStats: Accessor<WSStats>;
}> = (props) => {
  const st = () => props.latestStatus();
  const pred = () => props.latestPrediction();
  const tray = () => props.latestTrayState();
  const sug = () => props.activeSuggestion();
  const stats = () => props.wsStats();

  const transitionPct = createMemo(() => {
    const s = st();
    if (!s?.candidate_app || !s.transition_threshold_s) return null;
    return Math.min(
      100,
      Math.round((s.candidate_duration_s / s.transition_threshold_s) * 100),
    );
  });

  const appCounts = createMemo(() => {
    const s = st();
    if (!s?.last_app_counts) return [];
    return Object.entries(s.last_app_counts).sort(([, a], [, b]) => b - a);
  });

  return (
    <div
      style={{
        background: "var(--surface)",
        border: "1px solid #2a2a2a",
        "border-radius": "8px",
        padding: "6px 8px",
        "font-family": "'SF Mono', 'Fira Code', 'Cascadia Code', monospace",
        "font-size": "0.65rem",
        "min-width": "230px",
        color: "#d0d0d0",
        "box-shadow": "0 8px 32px rgba(0, 0, 0, 0.6)",
      }}
    >
      <div
        style={{
          "font-size": "0.75rem",
          "font-weight": "700",
          color: "#d0d0d0",
          "margin-bottom": "6px",
          "padding-bottom": "4px",
          "border-bottom": "1px solid #2a2a2a",
          "letter-spacing": "0.02em",
        }}
      >
        State Panel
      </div>

      <Section title="WebSocket">
        <Row
          label="status"
          value={props.status()}
          color={dotColor(props.status())}
        />
        <Row
          label="messages"
          value={`${stats().messageCount} total`}
          dim
        />
        <Row
          label="breakdown"
          value={`st:${stats().statusCount} pred:${stats().predictionCount} tray:${stats().trayStateCount} sug:${stats().suggestionCount}`}
          dim
          mono
        />
        <Row
          label="last_received"
          value={formatTime(stats().lastMessageAt)}
          dim
        />
        <Row
          label="reconnects"
          value={String(stats().reconnectCount)}
          dim
        />
        <Show when={stats().connectedSince}>
          <Row
            label="connected_since"
            value={formatTime(stats().connectedSince)}
            dim
          />
        </Show>
      </Section>

      <Section title="ActivityWatch">
        <Row
          label="connection"
          value={st()?.aw_connected ? "connected" : "disconnected"}
          color={st()?.aw_connected ? "#22c55e" : "#ef4444"}
        />
        <Row label="host" value={st()?.aw_host ?? "—"} dim mono />
        <Row
          label="bucket_id"
          value={st()?.aw_bucket_id ?? "—"}
          dim
          mono
        />
        <Row
          label="last_events"
          value={st() ? String(st()!.last_event_count) : "—"}
          dim
        />
        <Show when={appCounts().length > 0}>
          <div
            style={{
              "margin-top": "2px",
              "padding-left": "2px",
            }}
          >
            <span
              style={{
                color: "#7a7a7a",
                "font-size": "0.58rem",
                "text-transform": "uppercase",
              }}
            >
              app distribution (last poll)
            </span>
            <For each={appCounts()}>
              {([app, count]) => (
                <Row
                  label={`  ${app}`}
                  value={String(count)}
                  dim
                  mono
                />
              )}
            </For>
          </div>
        </Show>
      </Section>

      <Section title="Activity Monitor">
        <Row label="state" value={st()?.state ?? "—"} />
        <Row label="current_app" value={st()?.current_app ?? "—"} />
        <Row
          label="since"
          value={formatTime(st()?.current_app_since)}
          dim
        />
        <Row
          label="poll_interval"
          value={st() ? `${st()!.poll_seconds}s` : "—"}
          dim
        />
        <Row
          label="poll_count"
          value={st() ? String(st()!.poll_count) : "—"}
          dim
        />
        <Row
          label="last_poll"
          value={formatTime(st()?.last_poll_ts)}
          dim
        />
        <Row
          label="uptime"
          value={st() ? formatDuration(st()!.uptime_s) : "—"}
          dim
        />
        <Show when={st()?.candidate_app}>
          <Row
            label="candidate_app"
            value={st()!.candidate_app!}
            color="#eab308"
          />
          <Row
            label="candidate_progress"
            value={`${formatDuration(st()!.candidate_duration_s)} / ${formatDuration(st()!.transition_threshold_s)} (${transitionPct()}%)`}
            color="#eab308"
          />
          <ProgressBar pct={transitionPct()!} />
        </Show>
        <Show when={!st()?.candidate_app}>
          <Row label="candidate_app" value="none" dim />
        </Show>
      </Section>

      <Section title="Last Prediction">
        <Show
          when={pred()}
          fallback={<Row label="status" value="no prediction yet" dim />}
        >
          <Row
            label="label"
            value={pred()!.label}
            color={LABEL_COLORS[pred()!.label] ?? "#d0d0d0"}
          />
          <Row
            label="mapped_label"
            value={pred()!.mapped_label}
            color={LABEL_COLORS[pred()!.mapped_label] ?? "#d0d0d0"}
          />
          <Row
            label="confidence"
            value={`${Math.round(pred()!.confidence * 100)}%`}
            color={pred()!.confidence >= 0.5 ? "#22c55e" : "#ef4444"}
          />
          <Row label="ts" value={formatTime(pred()!.ts)} dim />
          <Show when={pred()!.current_app}>
            <Row label="trigger_app" value={pred()!.current_app!} dim />
          </Show>
        </Show>
      </Section>

      <Section title="Model">
        <Row
          label="loaded"
          value={tray() ? (tray()!.model_loaded ? "yes" : "no") : "—"}
          color={tray()?.model_loaded ? "#22c55e" : "#ef4444"}
        />
        <Show when={tray()?.model_dir}>
          <Row
            label="model_dir"
            value={truncPath(tray()!.model_dir)}
            dim
            mono
          />
        </Show>
        <Show when={tray()?.model_schema_hash}>
          <Row
            label="schema_hash"
            value={tray()!.model_schema_hash!}
            dim
            mono
          />
        </Show>
        <Show when={tray()?.suggested_label}>
          <Row
            label="suggested"
            value={tray()!.suggested_label!}
            color={LABEL_COLORS[tray()!.suggested_label!] ?? "#d0d0d0"}
          />
          <Row
            label="suggestion_conf"
            value={`${Math.round((tray()!.suggested_confidence ?? 0) * 100)}%`}
          />
        </Show>
        <Show when={tray() && !tray()!.suggested_label}>
          <Row label="suggested" value="none" dim />
        </Show>
      </Section>

      <Section title="Transitions">
        <Row
          label="total"
          value={tray() ? String(tray()!.transition_count) : "—"}
        />
        <Show
          when={tray()?.last_transition}
          fallback={<Row label="last" value="none yet" dim />}
        >
          <Row
            label="prev → new"
            value={`${tray()!.last_transition!.prev_app} → ${tray()!.last_transition!.new_app}`}
          />
          <Row
            label="block"
            value={`${formatTime(tray()!.last_transition!.block_start)} → ${formatTime(tray()!.last_transition!.block_end)}`}
            dim
          />
          <Row
            label="fired_at"
            value={formatTime(tray()!.last_transition!.fired_at)}
            dim
          />
        </Show>
      </Section>

      <Show when={sug()}>
        <Section title="Active Suggestion">
          <Row
            label="suggested"
            value={sug()!.suggested}
            color={LABEL_COLORS[sug()!.suggested] ?? "#d0d0d0"}
          />
          <Row
            label="confidence"
            value={`${Math.round(sug()!.confidence * 100)}%`}
          />
          <Row label="reason" value={sug()!.reason} dim />
          <Row label="old_label" value={sug()!.old_label} dim />
          <Row
            label="block"
            value={`${formatTime(sug()!.block_start)} → ${formatTime(sug()!.block_end)}`}
            dim
          />
        </Section>
      </Show>

      <Section title="Config">
        <Row
          label="data_dir"
          value={truncPath(tray()?.data_dir)}
          dim
          mono
        />
        <Row
          label="ui_port"
          value={tray() ? String(tray()!.ui_port) : "—"}
          dim
        />
        <Row
          label="dev_mode"
          value={tray() ? (tray()!.dev_mode ? "yes" : "no") : "—"}
          dim
        />
        <Row
          label="labels_saved"
          value={tray() ? String(tray()!.labels_saved_count) : "—"}
        />
      </Section>
    </div>
  );
};
