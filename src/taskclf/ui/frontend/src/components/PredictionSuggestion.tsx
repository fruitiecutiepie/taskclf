import {
  type Accessor,
  type Component,
  createEffect,
  createResource,
  createSignal,
  For,
  on,
  Show,
} from "solid-js";
import { core_labels_list, notification_accept, notification_skip } from "../lib/api";
import { time_format } from "../lib/format";
import { LABEL_COLORS } from "../lib/labelColors";
import { frontend_log_error } from "../lib/log";
import { overwrite_pending_from_api_error } from "../lib/overwrite_pending_from_api_error";
import {
  type LabelSuggestion,
  label_suggestion_key,
  type SuggestionClearReason,
} from "../lib/ws";
import { ActivitySummary } from "./ActivitySummary";
import { ErrorBanner } from "./ErrorBanner";
import { LabelOverwrite, type OverwritePending } from "./LabelOverwrite";

/** Matches `LabelOverwrite` / recorder control sizing. */
const btn_base = {
  padding: "2px 10px",
  "border-radius": "6px",
  cursor: "pointer" as const,
  "font-size": "0.7rem",
  "font-weight": "600",
};

function label_color(name: string, fallback = "var(--text)"): string {
  const color = LABEL_COLORS[name];
  if (color) {
    return color;
  }
  return fallback;
}

function suggestion_range_part_format(d: Date): string {
  return d.toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function suggestion_range_format(block_start: string, block_end: string): string {
  const start = new Date(block_start);
  const end = new Date(block_end);

  const crosses_local_day =
    start.getFullYear() !== end.getFullYear()
    || start.getMonth() !== end.getMonth()
    || start.getDate() !== end.getDate();

  if (!crosses_local_day) {
    return `${suggestion_range_part_format(start)} → ${time_format(block_end)}`;
  }

  return `${suggestion_range_part_format(start)} → ${suggestion_range_part_format(end)}`;
}

export const PredictionSuggestion: Component<{
  suggestion: Accessor<LabelSuggestion | null>;
  suggestions?: Accessor<LabelSuggestion[]>;
  on_saved?: () => void;
  on_dismiss?: (
    reason?: SuggestionClearReason,
    suggestion?: LabelSuggestion | null,
  ) => void;
  on_select?: (suggestion: LabelSuggestion) => void;
}> = (props) => {
  const s = () => props.suggestion();
  const [error, set_error] = createSignal<string | null>(null);
  const [busy, set_busy] = createSignal(false);
  const [overwrite_pending, set_overwrite_pending] =
    createSignal<OverwritePending | null>(null);
  const [change_label_open, set_change_label_open] = createSignal(false);
  const [selected_label, set_selected_label] = createSignal("");
  const [labels] = createResource(core_labels_list);

  const label_choices = () => {
    const loaded = labels();
    if (!loaded) {
      return [];
    }
    return loaded;
  };

  const pending_suggestions = () => {
    if (props.suggestions) {
      return props.suggestions();
    }
    const sg = s();
    if (!sg) {
      return [];
    }
    return [sg];
  };

  const active_key = () => {
    const sg = s();
    if (!sg) {
      return null;
    }
    return label_suggestion_key(sg);
  };

  const active_index = () => {
    const key = active_key();
    if (!key) {
      return -1;
    }
    return pending_suggestions().findIndex(
      (item) => label_suggestion_key(item) === key,
    );
  };

  const pending_count = () => pending_suggestions().length;

  const suggestion_position = () => {
    const idx = active_index();
    if (idx < 0) {
      return 0;
    }
    return idx + 1;
  };

  const correction_label = () => selected_label();

  function notify_saved() {
    if (props.on_saved) {
      props.on_saved();
    }
  }

  function notify_dismiss(
    reason: SuggestionClearReason,
    suggestion: LabelSuggestion | null,
  ) {
    if (props.on_dismiss) {
      props.on_dismiss(reason, suggestion);
    }
  }

  function select_suggestion(item: LabelSuggestion) {
    if (props.on_select) {
      props.on_select(item);
    }
  }

  function reset_for_suggestion(sg: LabelSuggestion) {
    set_overwrite_pending(null);
    set_change_label_open(false);
    set_selected_label(sg.suggested);
    set_error(null);
  }

  createEffect(
    on(
      () => {
        const sg = s();
        if (!sg) {
          return "";
        }
        return `${sg.block_start}:${sg.block_end}:${sg.suggested}`;
      },
      () => {
        const sg = s();
        if (!sg) {
          set_selected_label("");
          return;
        }
        reset_for_suggestion(sg);
      },
    ),
  );

  async function suggestion_save(label: string) {
    const sg = s();
    if (!sg || busy() || !label) {
      return;
    }
    set_busy(true);
    set_error(null);
    try {
      const payload = {
        block_start: sg.block_start,
        block_end: sg.block_end,
        label,
        ...(sg.suggestion_id ? { suggestion_id: sg.suggestion_id } : {}),
      };
      await notification_accept(payload);
      set_overwrite_pending(null);
      set_change_label_open(false);
      notify_saved();
      notify_dismiss("label_saved", sg);
    } catch (err: unknown) {
      const pending = overwrite_pending_from_api_error(err, {
        label,
        start: sg.block_start,
        end: sg.block_end,
        confidence: sg.confidence,
        extend_forward: false,
      });
      if (pending) {
        set_overwrite_pending(pending);
        return;
      }
      const msg = err instanceof Error ? err.message : "Failed to save inferred label";
      frontend_log_error("Failed to save inferred label", err);
      set_error(msg);
    } finally {
      set_busy(false);
    }
  }

  async function suggestion_accept() {
    const sg = s();
    if (!sg) {
      return;
    }
    await suggestion_save(sg.suggested);
  }

  async function overwrite_confirm() {
    const pending = overwrite_pending();
    if (!pending || busy()) {
      return;
    }
    set_busy(true);
    set_error(null);
    try {
      const sg = s();
      const payload = {
        block_start: pending.start,
        block_end: pending.end,
        label: pending.label,
        overwrite: true as const,
        ...(sg && sg.suggestion_id ? { suggestion_id: sg.suggestion_id } : {}),
      };
      await notification_accept(payload);
      set_overwrite_pending(null);
      notify_saved();
      notify_dismiss("label_saved", sg);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "overwrite failed";
      frontend_log_error("Failed to overwrite with suggested label", err);
      set_error(msg);
    } finally {
      set_busy(false);
    }
  }

  async function keep_all_confirm() {
    const pending = overwrite_pending();
    if (!pending || busy()) {
      return;
    }
    set_busy(true);
    set_error(null);
    try {
      const sg = s();
      const payload = {
        block_start: pending.start,
        block_end: pending.end,
        label: pending.label,
        allow_overlap: true as const,
        ...(sg && sg.suggestion_id ? { suggestion_id: sg.suggestion_id } : {}),
      };
      await notification_accept(payload);
      set_overwrite_pending(null);
      notify_saved();
      notify_dismiss("label_saved", sg);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "keep all failed";
      frontend_log_error("Failed to keep all with suggested label", err);
      set_error(msg);
    } finally {
      set_busy(false);
    }
  }

  async function suggestion_dismiss() {
    if (busy()) {
      return;
    }
    const sg = s();
    set_busy(true);
    set_error(null);
    try {
      if (sg && sg.suggestion_id) {
        await notification_skip({ suggestion_id: sg.suggestion_id });
      } else {
        await notification_skip(undefined);
      }
      set_overwrite_pending(null);
      set_change_label_open(false);
      notify_dismiss("skipped", sg);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Failed to dismiss suggestion";
      frontend_log_error("Failed to dismiss suggestion", err);
      set_error(msg);
    } finally {
      set_busy(false);
    }
  }

  function open_change_label(sg: LabelSuggestion) {
    set_error(null);
    set_selected_label(sg.suggested);
    set_change_label_open(true);
  }

  function cancel_change_label(sg: LabelSuggestion) {
    set_change_label_open(false);
    set_selected_label(sg.suggested);
  }

  const busy_opacity = () => (busy() ? "0.6" : "1");
  const busy_cursor = () => (busy() ? "not-allowed" : "pointer");

  const suggestion_time_range = () => {
    const sg = s();
    if (!sg) {
      return null;
    }
    return { start: sg.block_start, end: sg.block_end };
  };

  return (
    <Show when={s()}>
      {(active) => (
        <div
          style={{
            background:
              "linear-gradient(135deg, color-mix(in srgb, var(--warning) 16%, transparent) 0%, var(--surface) 55%)",
            border: "1px solid color-mix(in srgb, var(--warning) 42%, var(--border))",
            "border-left": "4px solid var(--warning)",
            "border-radius": "var(--radius)",
            padding: "10px 12px",
            "margin-bottom": "10px",
            "box-shadow":
              "0 0 0 1px color-mix(in srgb, var(--warning) 22%, transparent), 0 4px 18px color-mix(in srgb, var(--warning) 14%, transparent)",
            display: "flex",
            "flex-direction": "column",
            gap: "8px",
          }}
        >
          <Show when={pending_count() > 1}>
            <div
              style={{
                display: "flex",
                "flex-direction": "column",
                gap: "6px",
              }}
            >
              <div
                style={{
                  display: "flex",
                  "justify-content": "space-between",
                  "align-items": "center",
                  gap: "8px",
                  color: "var(--text-muted)",
                  "font-size": "0.68rem",
                  "font-weight": "700",
                  "letter-spacing": "0.02em",
                  "text-transform": "uppercase",
                }}
              >
                <span>Model suggestions ({pending_count()} pending)</span>
                <span>
                  {suggestion_position()} of {pending_count()}
                </span>
              </div>
              <ul
                aria-label="Pending suggestions"
                style={{
                  display: "grid",
                  "grid-template-columns": "repeat(auto-fit, minmax(96px, 1fr))",
                  gap: "4px",
                  "list-style": "none",
                  margin: "0",
                  padding: "0",
                }}
              >
                <For each={pending_suggestions()}>
                  {(item, index) => {
                    const is_active = () => label_suggestion_key(item) === active_key();
                    return (
                      <li>
                        <button
                          type="button"
                          aria-current={is_active() ? "true" : undefined}
                          aria-label={`Review suggestion ${index() + 1}: ${item.suggested}`}
                          disabled={busy()}
                          onClick={() => select_suggestion(item)}
                          style={{
                            border: is_active()
                              ? "1.5px solid var(--warning)"
                              : "1px solid var(--border)",
                            background: is_active()
                              ? "color-mix(in srgb, var(--warning) 18%, var(--surface))"
                              : "var(--surface)",
                            color: label_color(item.suggested),
                            "border-radius": "6px",
                            padding: "5px 6px",
                            cursor: busy() ? "not-allowed" : "pointer",
                            "text-align": "left",
                            "font-size": "0.62rem",
                            opacity: busy() ? "0.6" : "1",
                            width: "100%",
                          }}
                        >
                          <span
                            style={{
                              display: "block",
                              color: "var(--text-muted)",
                              "font-size": "0.55rem",
                              "margin-bottom": "1px",
                            }}
                          >
                            {suggestion_range_format(item.block_start, item.block_end)}
                          </span>
                          <span style={{ "font-weight": "700" }}>{item.suggested}</span>
                        </button>
                      </li>
                    );
                  }}
                </For>
              </ul>
            </div>
          </Show>
          <div
            style={{
              display: "flex",
              "flex-wrap": "wrap",
              "align-items": "flex-start",
              "justify-content": "space-between",
              gap: "8px",
            }}
          >
            <div style={{ "min-width": "0", flex: "1 1 140px" }}>
              <div style={{ "font-size": "0.8rem", "line-height": "1.35" }}>
                <span style={{ color: "var(--text-muted)" }}>
                  Model suggests a change:{" "}
                </span>
                <span
                  style={{
                    color: label_color(active().old_label),
                    "font-weight": "700",
                  }}
                >
                  {active().old_label}
                </span>
                <span style={{ color: "var(--text-muted)" }}> → </span>
                <span
                  style={{
                    color: label_color(active().suggested),
                    "font-weight": "700",
                  }}
                >
                  {active().suggested}
                </span>
                <span style={{ color: "var(--text-muted)" }}>
                  {" "}
                  ({Math.round(active().confidence * 100)}%)
                </span>
              </div>
              <div
                style={{
                  color: "var(--text-muted)",
                  "font-size": "0.7rem",
                  "margin-top": "4px",
                }}
              >
                {suggestion_range_format(active().block_start, active().block_end)}
              </div>
            </div>
            <div
              style={{
                display: "flex",
                "flex-wrap": "wrap",
                gap: "6px",
                "flex-shrink": "0",
                "align-items": "center",
              }}
            >
              <button
                type="button"
                disabled={busy()}
                onClick={suggestion_accept}
                style={{
                  ...btn_base,
                  border: "1px solid var(--accent, #6366f1)",
                  background: "var(--accent, #6366f1)",
                  color: "#fff",
                  opacity: busy_opacity(),
                  cursor: busy_cursor(),
                }}
              >
                Use suggestion
              </button>
              <button
                type="button"
                disabled={busy()}
                onClick={() => open_change_label(active())}
                style={{
                  ...btn_base,
                  border: "1px solid var(--border)",
                  background: "var(--surface)",
                  color: "var(--text)",
                  opacity: busy_opacity(),
                  cursor: busy_cursor(),
                }}
              >
                Change label
              </button>
              <button
                type="button"
                disabled={busy()}
                onClick={suggestion_dismiss}
                style={{
                  ...btn_base,
                  border: "1px solid var(--border)",
                  background: "var(--surface)",
                  color: "var(--text-muted)",
                  opacity: busy_opacity(),
                  cursor: busy_cursor(),
                }}
              >
                Skip
              </button>
            </div>
          </div>
          <Show when={change_label_open()}>
            <div
              style={{
                border: "1px solid var(--border)",
                "border-radius": "var(--radius)",
                background: "color-mix(in srgb, var(--surface) 92%, var(--warning))",
                padding: "8px",
                display: "flex",
                "flex-direction": "column",
                gap: "8px",
              }}
            >
              <div
                style={{
                  color: "var(--text-muted)",
                  "font-size": "0.72rem",
                  "line-height": "1.35",
                }}
              >
                What should this time block be labeled instead?
              </div>
              <Show
                when={!labels.loading}
                fallback={
                  <div
                    style={{
                      color: "var(--text-muted)",
                      "font-size": "0.68rem",
                    }}
                  >
                    Loading label choices...
                  </div>
                }
              >
                <fieldset
                  aria-label="Choose replacement label"
                  style={{
                    display: "grid",
                    "grid-template-columns": "repeat(4, minmax(0, 1fr))",
                    gap: "4px",
                    border: "0",
                    margin: "0",
                    padding: "0",
                  }}
                >
                  <For each={label_choices()}>
                    {(label_name) => {
                      const is_selected = () => label_name === correction_label();
                      const is_suggested = () => label_name === active().suggested;
                      return (
                        <button
                          type="button"
                          aria-pressed={is_selected()}
                          disabled={busy()}
                          onClick={() => set_selected_label(label_name)}
                          style={{
                            padding: "5px 3px",
                            "border-radius": "5px",
                            border: is_selected()
                              ? `1.5px solid ${label_color(label_name, "var(--accent, #6366f1)")}`
                              : is_suggested()
                                ? "1px dashed var(--warning)"
                                : "1px solid var(--border)",
                            background: is_selected()
                              ? "color-mix(in srgb, var(--accent, #6366f1) 14%, var(--surface))"
                              : "var(--surface)",
                            color: label_color(label_name),
                            cursor: busy_cursor(),
                            "font-size": "0.62rem",
                            "font-weight": is_selected() ? "700" : "600",
                            "text-align": "center",
                            opacity: busy()
                              ? "0.5"
                              : is_selected()
                                ? "1"
                                : is_suggested()
                                  ? "0.9"
                                  : "0.78",
                          }}
                        >
                          <span>{label_name}</span>
                          <Show when={is_suggested()}>
                            <span
                              style={{
                                display: "block",
                                color: "var(--text-muted)",
                                "font-size": "0.5rem",
                                "font-weight": "600",
                                "margin-top": "1px",
                              }}
                            >
                              suggested
                            </span>
                          </Show>
                        </button>
                      );
                    }}
                  </For>
                </fieldset>
              </Show>
              <div
                style={{
                  display: "flex",
                  "justify-content": "flex-end",
                  gap: "6px",
                  "flex-wrap": "wrap",
                }}
              >
                <button
                  type="button"
                  disabled={busy()}
                  onClick={() => cancel_change_label(active())}
                  style={{
                    ...btn_base,
                    border: "1px solid var(--border)",
                    background: "var(--surface)",
                    color: "var(--text-muted)",
                    opacity: busy_opacity(),
                    cursor: busy_cursor(),
                  }}
                >
                  Cancel
                </button>
                <button
                  type="button"
                  disabled={busy() || !correction_label()}
                  onClick={() => suggestion_save(correction_label())}
                  style={{
                    ...btn_base,
                    border: "1px solid var(--accent, #6366f1)",
                    background: "var(--accent, #6366f1)",
                    color: "#fff",
                    opacity: busy_opacity(),
                    cursor: busy() || !correction_label() ? "not-allowed" : "pointer",
                  }}
                >
                  Save as {correction_label()}
                </button>
              </div>
            </div>
          </Show>
          <ActivitySummary time_range={suggestion_time_range} show_empty />
          <Show when={overwrite_pending()}>
            {(pending) => (
              <LabelOverwrite
                pending={pending()}
                on_confirm={overwrite_confirm}
                on_keep_all={keep_all_confirm}
                on_cancel={() => set_overwrite_pending(null)}
              />
            )}
          </Show>
          <Show when={error()}>
            {(message) => (
              <ErrorBanner message={message()} on_close={() => set_error(null)} />
            )}
          </Show>
        </div>
      )}
    </Show>
  );
};
