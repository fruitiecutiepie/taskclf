import {
  type Accessor,
  type Component,
  For,
  Show,
  createMemo,
  createSignal,
} from "solid-js";
import type {
  ConnectionStatus,
  LabelSuggestion,
  Prediction,
  StatusEvent,
  TrainState,
  TrayState,
  WSStats,
} from "../lib/ws";
import { LABEL_COLORS } from "../lib/labelColors";
import { dotColor } from "../lib/labelColors";
import { formatDuration, formatTime, truncPath } from "../lib/format";
import { LabelHistory } from "./LabelHistory";
import { TrainingPanel } from "./TrainingPanel";
import { StatusRow } from "./ui/StatusRow";
import { StatusSection } from "./ui/StatusSection";
import { StatusProgress } from "./ui/StatusProgress";

type PanelTab = "system" | "history" | "training";

export const StatusPanel: Component<{
  status: Accessor<ConnectionStatus>;
  latestStatus: Accessor<StatusEvent | null>;
  latestPrediction: Accessor<Prediction | null>;
  latestTrayState: Accessor<TrayState | null>;
  activeSuggestion: Accessor<LabelSuggestion | null>;
  wsStats: Accessor<WSStats>;
  trainState: Accessor<TrainState>;
}> = (props) => {
  const [tab, setTab] = createSignal<PanelTab>("system");
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

  const activitySummary = createMemo(() => {
    const s = st();
    if (!s) return "—";
    return s.current_app ? `${s.state} · ${s.current_app}` : s.state;
  });

  const predSummary = createMemo(() => {
    const p = pred();
    if (!p) return "none";
    return `${p.mapped_label} ${Math.round(p.confidence * 100)}%`;
  });
  const predSummaryColor = createMemo(() => {
    const p = pred();
    return p ? (LABEL_COLORS[p.mapped_label] ?? "#e0e0e0") : "#a0a0a0";
  });

  const modelSummary = createMemo(() => {
    const t = tray();
    if (!t) return "—";
    return t.model_loaded ? "loaded" : "not loaded";
  });
  const modelSummaryColor = createMemo(() => {
    const t = tray();
    return t?.model_loaded ? "#22c55e" : "#ef4444";
  });

  const transitionSummary = createMemo(() => {
    const t = tray();
    return t ? String(t.transition_count) : "—";
  });

  const sugSummary = createMemo(() => {
    const s = sug();
    if (!s) return "";
    return `${s.suggested} ${Math.round(s.confidence * 100)}%`;
  });
  const sugSummaryColor = createMemo(() => {
    const s = sug();
    return s ? (LABEL_COLORS[s.suggested] ?? "#e0e0e0") : "#a0a0a0";
  });

  const awSummary = createMemo(() =>
    st()?.aw_connected ? "connected" : "disconnected",
  );
  const awSummaryColor = createMemo(() =>
    st()?.aw_connected ? "#22c55e" : "#ef4444",
  );

  const wsSummary = createMemo(() => props.status());
  const wsSummaryColor = createMemo(() => dotColor(props.status()));

  const configSummary = createMemo(() => {
    const t = tray();
    if (!t) return "—";
    return t.dev_mode ? "dev" : "prod";
  });

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
      <div
        style={{
          display: "flex",
          "margin-bottom": "6px",
          "padding-bottom": "4px",
          "border-bottom": "1px solid #2a2a2a",
          gap: "0",
        }}
      >
        {(["system", "history", "training"] as PanelTab[]).map((t) => (
          <button
            onClick={() => setTab(t)}
            style={{
              flex: "1",
              padding: "3px 0",
              border: "none",
              background: tab() === t ? "#333" : "transparent",
              color: tab() === t ? "#e0e0e0" : "#9a9a9a",
              "font-size": "0.7rem",
              "font-weight": tab() === t ? "700" : "500",
              "font-family": "inherit",
              cursor: "pointer",
              "border-radius": "6px",
              "text-transform": "capitalize",
              "letter-spacing": "0.02em",
              transition: "all 0.15s ease",
            }}
          >
            {t}
          </button>
        ))}
      </div>

      <Show when={tab() === "system"}>
      <StatusSection
        title="Activity Monitor"
        summary={activitySummary()}
        defaultOpen
      >
        <StatusRow label="state" value={st()?.state ?? "—"} />
        <StatusRow label="current_app" value={st()?.current_app ?? "—"} />
        <StatusRow
          label="since"
          value={formatTime(st()?.current_app_since)}
          dim
        />
        <StatusRow
          label="poll_interval"
          value={st() ? `${st()!.poll_seconds}s` : "—"}
          dim
        />
        <StatusRow
          label="poll_count"
          value={st() ? String(st()!.poll_count) : "—"}
          dim
        />
        <StatusRow
          label="last_poll"
          value={formatTime(st()?.last_poll_ts)}
          dim
        />
        <StatusRow
          label="uptime"
          value={st() ? formatDuration(st()!.uptime_s) : "—"}
          dim
        />
        <Show when={st()?.candidate_app}>
          <StatusRow
            label="candidate_app"
            value={st()!.candidate_app!}
            color="#eab308"
          />
          <StatusRow
            label="candidate_progress"
            value={`${formatDuration(st()!.candidate_duration_s)} / ${formatDuration(st()!.transition_threshold_s)} (${transitionPct()}%)`}
            color="#eab308"
          />
          <StatusProgress pct={transitionPct()!} />
        </Show>
        <Show when={!st()?.candidate_app}>
          <StatusRow label="candidate_app" value="none" dim />
        </Show>
      </StatusSection>

      <StatusSection
        title={pred()?.provenance === "manual" ? "Last Label" : "Last Prediction"}
        summary={predSummary()}
        summaryColor={predSummaryColor()}
        defaultOpen
      >
        <Show
          when={pred()}
          fallback={<StatusRow label="status" value="no prediction yet" dim />}
        >
          <StatusRow
            label="provenance"
            value={pred()!.provenance ?? "unknown"}
            dim
          />
          <StatusRow
            label="label"
            value={pred()!.label}
            color={LABEL_COLORS[pred()!.label] ?? "#e0e0e0"}
          />
          <StatusRow
            label="mapped_label"
            value={pred()!.mapped_label}
            color={LABEL_COLORS[pred()!.mapped_label] ?? "#e0e0e0"}
          />
          <StatusRow
            label="confidence"
            value={`${Math.round(pred()!.confidence * 100)}%`}
            color={pred()!.confidence >= 0.5 ? "#22c55e" : "#ef4444"}
          />
          <StatusRow label="ts" value={formatTime(pred()!.ts)} dim />
          <Show when={pred()!.current_app}>
            <StatusRow label="trigger_app" value={pred()!.current_app!} dim />
          </Show>
        </Show>
      </StatusSection>

      <StatusSection
        title="Model"
        summary={modelSummary()}
        summaryColor={modelSummaryColor()}
      >
        <StatusRow
          label="loaded"
          value={tray() ? (tray()!.model_loaded ? "yes" : "no") : "—"}
          color={tray()?.model_loaded ? "#22c55e" : "#ef4444"}
        />
        <Show when={tray()?.model_dir}>
          <StatusRow
            label="model_dir"
            value={truncPath(tray()!.model_dir)}
            dim
            mono
          />
        </Show>
        <Show when={tray()?.model_schema_hash}>
          <StatusRow
            label="schema_hash"
            value={tray()!.model_schema_hash!}
            dim
            mono
          />
        </Show>
        <Show when={tray()?.suggested_label}>
          <StatusRow
            label="suggested"
            value={tray()!.suggested_label!}
            color={LABEL_COLORS[tray()!.suggested_label!] ?? "#e0e0e0"}
          />
          <StatusRow
            label="suggestion_conf"
            value={`${Math.round((tray()!.suggested_confidence ?? 0) * 100)}%`}
          />
        </Show>
        <Show when={tray() && !tray()!.suggested_label}>
          <StatusRow label="suggested" value="none" dim />
        </Show>
      </StatusSection>

      <StatusSection title="Transitions" summary={transitionSummary()}>
        <StatusRow
          label="total"
          value={tray() ? String(tray()!.transition_count) : "—"}
        />
        <Show
          when={tray()?.last_transition}
          fallback={<StatusRow label="last" value="none yet" dim />}
        >
          <StatusRow
            label="prev → new"
            value={`${tray()!.last_transition!.prev_app} → ${tray()!.last_transition!.new_app}`}
          />
          <StatusRow
            label="block"
            value={`${formatTime(tray()!.last_transition!.block_start)} → ${formatTime(tray()!.last_transition!.block_end)}`}
            dim
          />
          <StatusRow
            label="fired_at"
            value={formatTime(tray()!.last_transition!.fired_at)}
            dim
          />
        </Show>
      </StatusSection>

      <Show when={sug()}>
        <StatusSection
          title="Active Suggestion"
          summary={sugSummary()}
          summaryColor={sugSummaryColor()}
        >
          <StatusRow
            label="suggested"
            value={sug()!.suggested}
            color={LABEL_COLORS[sug()!.suggested] ?? "#e0e0e0"}
          />
          <StatusRow
            label="confidence"
            value={`${Math.round(sug()!.confidence * 100)}%`}
          />
          <StatusRow label="reason" value={sug()!.reason} dim />
          <StatusRow label="old_label" value={sug()!.old_label} dim />
          <StatusRow
            label="block"
            value={`${formatTime(sug()!.block_start)} → ${formatTime(sug()!.block_end)}`}
            dim
          />
        </StatusSection>
      </Show>

      <StatusSection
        title="ActivityWatch"
        summary={awSummary()}
        summaryColor={awSummaryColor()}
      >
        <StatusRow
          label="connection"
          value={st()?.aw_connected ? "connected" : "disconnected"}
          color={st()?.aw_connected ? "#22c55e" : "#ef4444"}
        />
        <StatusRow label="host" value={st()?.aw_host ?? "—"} dim mono />
        <StatusRow
          label="bucket_id"
          value={st()?.aw_bucket_id ?? "—"}
          dim
          mono
        />
        <StatusRow
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
                color: "#9a9a9a",
                "font-size": "0.58rem",
                "text-transform": "uppercase",
              }}
            >
              app distribution (last poll)
            </span>
            <For each={appCounts()}>
              {([app, count]) => (
                <StatusRow
                  label={`  ${app}`}
                  value={String(count)}
                  dim
                  mono
                />
              )}
            </For>
          </div>
        </Show>
      </StatusSection>

      <StatusSection
        title="WebSocket"
        summary={wsSummary()}
        summaryColor={wsSummaryColor()}
      >
        <StatusRow
          label="status"
          value={props.status()}
          color={dotColor(props.status())}
        />
        <StatusRow
          label="messages"
          value={`${stats().messageCount} total`}
          dim
        />
        <StatusRow
          label="breakdown"
          value={`st:${stats().statusCount} pred:${stats().predictionCount} tray:${stats().trayStateCount} sug:${stats().suggestionCount}`}
          dim
          mono
        />
        <StatusRow
          label="last_received"
          value={formatTime(stats().lastMessageAt)}
          dim
        />
        <StatusRow
          label="reconnects"
          value={String(stats().reconnectCount)}
          dim
        />
        <Show when={stats().connectedSince}>
          <StatusRow
            label="connected_since"
            value={formatTime(stats().connectedSince)}
            dim
          />
        </Show>
      </StatusSection>

      <StatusSection title="Config" summary={configSummary()}>
        <StatusRow
          label="data_dir"
          value={truncPath(tray()?.data_dir)}
          dim
          mono
        />
        <StatusRow
          label="ui_port"
          value={tray() ? String(tray()!.ui_port) : "—"}
          dim
        />
        <StatusRow
          label="dev_mode"
          value={tray() ? (tray()!.dev_mode ? "yes" : "no") : "—"}
          dim
        />
        <StatusRow
          label="labels_saved"
          value={tray() ? String(tray()!.labels_saved_count) : "—"}
        />
      </StatusSection>
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
