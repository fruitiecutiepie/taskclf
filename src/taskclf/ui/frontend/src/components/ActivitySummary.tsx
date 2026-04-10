import {
  type Accessor,
  type Component,
  createEffect,
  createSignal,
  For,
  onCleanup,
  Show,
} from "solid-js";
import {
  type ActivitySummary as ActivitySummaryData,
  activity_summary_get,
} from "../lib/api";
import type { TimeRange } from "../lib/date";
import { time_range_minutes } from "../lib/date";
import { app_name_short, rate_fmt } from "../lib/format";
import { LABEL_COLORS } from "../lib/labelColors";
import type { Prediction } from "../lib/ws";
import { ActivitySourceSetupCallout } from "./ActivitySourceSetupCallout";

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

  const [summary, set_summary] = createSignal<ActivitySummaryData | null>(null);
  const [is_loading, set_is_loading] = createSignal(false);
  const [request_failed, set_request_failed] = createSignal(false);

  createEffect(() => {
    const r = range();
    if (!r) {
      set_summary(null);
      set_is_loading(false);
      set_request_failed(false);
      return;
    }

    let cancelled = false;
    set_is_loading(true);
    set_request_failed(false);

    void activity_summary_get(r.start, r.end)
      .then((result) => {
        if (cancelled) {
          return;
        }
        set_summary(result);
        set_is_loading(false);
      })
      .catch(() => {
        if (cancelled) {
          return;
        }
        set_summary(null);
        set_request_failed(true);
        set_is_loading(false);
      });

    onCleanup(() => {
      cancelled = true;
    });
  });

  const pred = () => props.prediction?.();
  const provider = () => summary()?.activity_provider ?? null;
  const recent_apps = () => (summary()?.recent_apps ?? []).slice(0, 3);
  const has_recent_apps = () => recent_apps().length > 0;
  const feature_apps = () => (summary()?.top_apps ?? []).slice(0, 5);
  const has_feature_apps = () => feature_apps().length > 0;
  const has_apps = () => has_recent_apps() || has_feature_apps();
  const has_stats = () => {
    const s = summary();
    return s && (s.mean_keys_per_min != null || s.mean_clicks_per_min != null);
  };
  const has_coverage = () => {
    const s = summary();
    return s && s.total_buckets > 0;
  };
  const has_anything = () =>
    pred()
    || summary()?.range_state === "provider_unavailable"
    || has_apps()
    || has_stats()
    || has_coverage();
  const should_show = () => has_anything() || props.show_empty || request_failed();

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
          when={!request_failed()}
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
                Activity summary unavailable right now
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
                    {summary()?.message ?? "No activity data for this window"}
                  </div>
                </div>
              </Show>
            }
          >
            <div style={container_style}>
              <Show when={pred()}>{(p) => <PredictionBadge p={p} />}</Show>

              <Show
                when={summary()?.range_state === "provider_unavailable" && provider()}
              >
                {(status) => <ActivitySourceSetupCallout provider={status()} compact />}
              </Show>

              <Show when={summary()?.range_state === "ok"}>
                <>
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
                        when={has_recent_apps()}
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
                                <span
                                  style={{ "font-weight": "600", color: "var(--text)" }}
                                >
                                  {app_name_short(entry.app_id)}
                                </span>
                                {entry.buckets}m
                              </span>
                            )}
                          </For>
                        }
                      >
                        <For each={recent_apps()}>
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
                              <span
                                style={{ "font-weight": "600", color: "var(--text)" }}
                              >
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
                      <Show when={rate_fmt(summary()?.mean_keys_per_min ?? null)}>
                        {(v) => <span>keys {v()}/m</span>}
                      </Show>
                      <Show when={rate_fmt(summary()?.mean_clicks_per_min ?? null)}>
                        {(v) => <span>clicks {v()}/m</span>}
                      </Show>
                      <Show when={rate_fmt(summary()?.mean_scroll_per_min ?? null)}>
                        {(v) => <span>scroll {v()}/m</span>}
                      </Show>
                      <Show when={has_coverage()}>
                        <span style={{ color: "var(--text-muted)", opacity: "0.7" }}>
                          {summary()?.total_buckets}m
                          <Show when={(summary()?.session_count ?? 0) > 1}>
                            {" "}
                            / {summary()?.session_count} sessions
                          </Show>
                        </span>
                      </Show>
                    </div>
                  </Show>
                </>
              </Show>
            </div>
          </Show>
        </Show>
      </Show>
    </Show>
  );
};
