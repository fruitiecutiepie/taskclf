import { type Accessor, type Component, createResource, For, Show } from "solid-js";
import { fetchAWLive, fetchFeatureSummary } from "../lib/api";
import type { Prediction } from "../lib/ws";
import { LABEL_COLORS } from "./StatePanel";

function shortAppName(app: string): string {
  const parts = app.split(".");
  return parts[parts.length - 1];
}

function fmtRate(v: number | null): string | null {
  if (v == null) return null;
  return v < 10 ? v.toFixed(1) : String(Math.round(v));
}

interface TimeRange {
  start: string;
  end: string;
}

function timeRangeForMinutes(mins: number): TimeRange | null {
  if (mins < 1) return null;
  const now = new Date();
  const start = new Date(now.getTime() - mins * 60_000);
  return {
    start: start.toISOString(),
    end: now.toISOString(),
  };
}

export const ActivityContext: Component<{
  minutes: Accessor<number>;
  prediction?: Accessor<Prediction | null>;
}> = (props) => {
  const range = () => timeRangeForMinutes(props.minutes());

  const [awApps] = createResource(range, async (r) => {
    if (!r) return [];
    try {
      return await fetchAWLive(r.start, r.end);
    } catch {
      return [];
    }
  });

  const [features] = createResource(range, async (r) => {
    if (!r) return null;
    try {
      return await fetchFeatureSummary(r.start, r.end);
    } catch {
      return null;
    }
  });

  const pred = () => props.prediction?.();
  const hasApps = () => (awApps() ?? []).length > 0;
  const hasStats = () => {
    const f = features();
    return f && (f.mean_keys_per_min != null || f.mean_clicks_per_min != null);
  };
  const hasAnything = () => pred() || hasApps() || hasStats();

  return (
    <Show when={hasAnything()}>
      <div
        style={{
          padding: "4px 0 6px",
          margin: "0 0 4px",
          "border-top": "1px dashed var(--border)",
          "border-bottom": "1px dashed var(--border)",
          display: "flex",
          "flex-direction": "column",
          gap: "3px",
        }}
      >
        {/* Model prediction */}
        <Show when={pred()}>
          <div
            style={{
              display: "flex",
              "align-items": "center",
              gap: "4px",
              "font-size": "0.65rem",
            }}
          >
            <span style={{ color: "var(--text-muted)" }}>Model:</span>
            <span
              style={{
                color: LABEL_COLORS[pred()!.mapped_label ?? pred()!.label] ?? "var(--text)",
                "font-weight": "600",
              }}
            >
              {pred()!.mapped_label || pred()!.label}
            </span>
            <span style={{ color: "var(--text-muted)" }}>
              ({Math.round(pred()!.confidence * 100)}%)
            </span>
          </div>
        </Show>

        {/* Top apps */}
        <Show when={hasApps()}>
          <div
            style={{
              display: "flex",
              "align-items": "center",
              gap: "6px",
              "flex-wrap": "wrap",
              "font-size": "0.6rem",
            }}
          >
            <For each={(awApps() ?? []).slice(0, 3)}>
              {(entry) => (
                <span
                  style={{
                    display: "inline-flex",
                    "align-items": "center",
                    gap: "2px",
                    padding: "1px 5px",
                    "border-radius": "8px",
                    background: "var(--surface)",
                    border: "1px solid var(--border)",
                    color: "var(--text-muted)",
                  }}
                >
                  <span style={{ "font-weight": "600", color: "var(--text)" }}>
                    {shortAppName(entry.app)}
                  </span>
                  {entry.events}
                </span>
              )}
            </For>
          </div>
        </Show>

        {/* Input stats */}
        <Show when={hasStats()}>
          <div
            style={{
              display: "flex",
              "align-items": "center",
              gap: "8px",
              "font-size": "0.6rem",
              color: "var(--text-muted)",
            }}
          >
            <Show when={fmtRate(features()!.mean_keys_per_min)}>
              {(v) => <span>keys {v()}/m</span>}
            </Show>
            <Show when={fmtRate(features()!.mean_clicks_per_min)}>
              {(v) => <span>clicks {v()}/m</span>}
            </Show>
            <Show when={fmtRate(features()!.mean_scroll_per_min)}>
              {(v) => <span>scroll {v()}/m</span>}
            </Show>
          </div>
        </Show>
      </div>
    </Show>
  );
};
