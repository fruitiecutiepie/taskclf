import { type Accessor, type Component, createResource, For, Show } from "solid-js";
import { aw_live_list, feature_summary_get } from "../lib/api";
import type { TimeRange } from "../lib/date";
import { time_range_minutes } from "../lib/date";
import { app_name_short, rate_fmt } from "../lib/format";
import { LABEL_COLORS } from "../lib/labelColors";
import type { Prediction } from "../lib/ws";

const PredictionBadge: Component<{ p: Accessor<Prediction> }> = (props) => (
  <div
    style={{
      display: "flex",
      "align-items": "center",
      gap: "4px",
      "font-size": "0.65rem",
    }}
  >
    <span style={{ color: "var(--text-muted)" }}>
      {props.p().provenance === "manual" ? "Label:" : "Model:"}
    </span>
    <span
      style={{
        color: LABEL_COLORS[props.p().mapped_label ?? props.p().label] ?? "var(--text)",
        "font-weight": "600",
      }}
    >
      {props.p().mapped_label || props.p().label}
    </span>
    <span style={{ color: "var(--text-muted)" }}>
      ({Math.round(props.p().confidence * 100)}%)
    </span>
  </div>
);

export const ActivitySummary: Component<{
  minutes?: Accessor<number>;
  time_range?: Accessor<TimeRange | null>;
  prediction?: Accessor<Prediction | null>;
  show_empty?: boolean;
}> = (props) => {
  const range = () =>
    props.time_range?.()
    ?? (props.minutes ? time_range_minutes(props.minutes()) : null);

  const [awApps] = createResource(range, async (r) => {
    if (!r) {
      return [];
    }
    try {
      return await aw_live_list(r.start, r.end);
    } catch {
      return [];
    }
  });

  const [features] = createResource(range, async (r) => {
    if (!r) {
      return null;
    }
    try {
      return await feature_summary_get(r.start, r.end);
    } catch {
      return null;
    }
  });

  const pred = () => props.prediction?.();
  const is_loading = () => awApps.loading || features.loading;
  const has_aw_apps = () => (awApps() ?? []).length > 0;
  const feature_apps = () => (features()?.top_apps ?? []).slice(0, 5);
  const has_feature_apps = () => feature_apps().length > 0;
  const has_apps = () => has_aw_apps() || has_feature_apps();
  const has_stats = () => {
    const f = features();
    return f && (f.mean_keys_per_min != null || f.mean_clicks_per_min != null);
  };
  const has_coverage = () => {
    const f = features();
    return f && f.total_buckets > 0;
  };
  const has_anything = () => pred() || has_apps() || has_stats() || has_coverage();
  const should_show = () => has_anything() || props.show_empty;

  const container_style = {
    padding: "4px 0 6px",
    margin: "0 0 4px",
    "border-top": "1px dashed var(--border)",
    "border-bottom": "1px dashed var(--border)",
    display: "flex",
    "flex-direction": "column",
    gap: "3px",
  } as const;

  return (
    <Show when={should_show()}>
      <Show
        when={!is_loading()}
        fallback={
          <div style={container_style}>
            <div
              style={{
                "font-size": "0.6rem",
                color: "var(--text-muted)",
                opacity: "0.6",
                "text-align": "center",
                padding: "2px 0",
              }}
            >
              Loading activity…
            </div>
          </div>
        }
      >
        <Show
          when={has_anything()}
          fallback={
            <Show when={props.show_empty}>
              <div style={container_style}>
                <div
                  style={{
                    "font-size": "0.6rem",
                    color: "var(--text-muted)",
                    opacity: "0.5",
                    "text-align": "center",
                    padding: "2px 0",
                  }}
                >
                  No activity data for this window
                </div>
              </div>
            </Show>
          }
        >
          <div style={container_style}>
            <Show when={pred()}>{(p) => <PredictionBadge p={p} />}</Show>

            <Show when={has_apps()}>
              <div
                style={{
                  display: "flex",
                  "align-items": "center",
                  gap: "6px",
                  "flex-wrap": "wrap",
                  "font-size": "0.6rem",
                }}
              >
                <Show
                  when={has_aw_apps()}
                  fallback={
                    <For each={feature_apps()}>
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
                            {app_name_short(entry.app_id)}
                          </span>
                          {entry.buckets}m
                        </span>
                      )}
                    </For>
                  }
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
                          {app_name_short(entry.app)}
                        </span>
                        {entry.events}
                      </span>
                    )}
                  </For>
                </Show>
              </div>
            </Show>

            <Show when={has_stats() || has_coverage()}>
              <div
                style={{
                  display: "flex",
                  "align-items": "center",
                  gap: "8px",
                  "font-size": "0.6rem",
                  color: "var(--text-muted)",
                  "flex-wrap": "wrap",
                }}
              >
                <Show when={rate_fmt(features()?.mean_keys_per_min ?? null)}>
                  {(v) => <span>keys {v()}/m</span>}
                </Show>
                <Show when={rate_fmt(features()?.mean_clicks_per_min ?? null)}>
                  {(v) => <span>clicks {v()}/m</span>}
                </Show>
                <Show when={rate_fmt(features()?.mean_scroll_per_min ?? null)}>
                  {(v) => <span>scroll {v()}/m</span>}
                </Show>
                <Show when={has_coverage()}>
                  <span style={{ color: "var(--text-muted)", opacity: "0.7" }}>
                    {features()?.total_buckets}m
                    <Show when={(features()?.session_count ?? 0) > 1}>
                      {" "}
                      / {features()?.session_count} sessions
                    </Show>
                  </span>
                </Show>
              </div>
            </Show>
          </div>
        </Show>
      </Show>
    </Show>
  );
};
