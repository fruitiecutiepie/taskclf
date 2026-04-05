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
import { core_labels_list, label_create, label_update, labels_list } from "../lib/api";
import { iso_date_parse } from "../lib/date";
import { label_overwrite_pending_upd_get } from "../lib/label_overwrite_pending_upd_get";
import { LABEL_COLORS } from "../lib/labelColors";
import { label_entry_is_open_ended } from "../lib/labelTimeline";
import { overwrite_pending_from_api_error } from "../lib/overwrite_pending_from_api_error";
import type { LabelSuggestion, Prediction } from "../lib/ws";
import { ActivitySummary } from "./ActivitySummary";
import { ErrorBanner } from "./ErrorBanner";
import { LabelConfidence } from "./LabelConfidence";
import { LabelExtendToggle } from "./LabelExtendToggle";
import { LabelFlash } from "./LabelFlash";
import { LabelLast } from "./LabelLast";
import { LabelOverwrite, type OverwritePending } from "./LabelOverwrite";
import { LabelTimePicker } from "./LabelTimePicker";
import { PredictionSuggestion } from "./PredictionSuggestion";

const EXTEND_FWD_KEY = "taskclf:extendForward";
const _LEGACY_KEY = "taskclf:extendPrevious";
function extend_forward_pref_read(): boolean {
  try {
    const v = localStorage.getItem(EXTEND_FWD_KEY) ?? localStorage.getItem(_LEGACY_KEY);
    return v !== "false";
  } catch {
    return true;
  }
}

type LabelRecorderProps = {
  max_height?: number;
  on_collapse: () => void;
  prediction?: Accessor<Prediction | null>;
  suggestion?: Accessor<LabelSuggestion | null>;
  label_change_count?: Accessor<number>;
  on_suggestion_dismiss?: () => void;
};

export const LabelRecorder: Component<LabelRecorderProps> = (props) => {
  const [labels] = createResource(core_labels_list);
  const [label_version, set_label_version] = createSignal(0);
  const [last_label] = createResource(
    () => [label_version(), props.label_change_count?.() ?? 0] as const,
    async () => {
      try {
        const rows = await labels_list(1);
        return rows.length ? rows[0] : null;
      } catch {
        return null;
      }
    },
  );
  const [flash, set_flash] = createSignal<string | null>(null);
  const [error, set_error] = createSignal<string | null>(null);
  const [overwrite_pending, set_overwrite_pending] =
    createSignal<OverwritePending | null>(null);
  const [selected_minutes, set_selected_minutes] = createSignal(0);
  const [extend_fwd, set_extend_fwd] = createSignal(extend_forward_pref_read());
  const [fill_from_last, set_fill_from_last] = createSignal(false);
  const [conf_percent, set_conf_percent] = createSignal(100);
  const [stop_current_pending, set_stop_current_pending] = createSignal(false);
  const [stop_current_busy, set_stop_current_busy] = createSignal(false);

  const current_label = () => {
    const row = last_label();
    return row && label_entry_is_open_ended(row) ? row : null;
  };

  createEffect(
    on(
      last_label,
      () => {
        if (overwrite_pending()) {
          set_overwrite_pending(null);
        }
        if (stop_current_pending()) {
          set_stop_current_pending(false);
        }
      },
      { defer: true },
    ),
  );

  createEffect(
    on(
      () => [selected_minutes(), fill_from_last()] as const,
      () => {
        const pending = overwrite_pending();
        if (!pending) {
          return;
        }

        const updated = label_overwrite_pending_upd_get(
          pending,
          {
            selected_minutes: selected_minutes(),
            fill_from_last: fill_from_last(),
            last_label_end_ts: last_label()?.end_ts ?? null,
            extend_fwd: extend_fwd(),
          },
          new Date(),
        );

        set_overwrite_pending(updated);
      },
      { defer: true },
    ),
  );

  function extend_fwd_toggle() {
    const next = !extend_fwd();
    set_extend_fwd(next);
    try {
      localStorage.setItem(EXTEND_FWD_KEY, String(next));
    } catch {}
  }

  async function label_now(label: string) {
    const mins = selected_minutes();
    const now = new Date();
    let start: Date;
    let force_extend_fwd = false;

    const last_end_ts = last_label()?.end_ts;
    if (fill_from_last() && last_end_ts) {
      start = iso_date_parse(last_end_ts);
    } else if (mins === 0) {
      start = now;
      force_extend_fwd = true;
    } else {
      start = new Date(now.getTime() - mins * 60_000);
    }
    const effective_extend = force_extend_fwd || extend_fwd();
    set_error(null);
    try {
      await label_create({
        start_ts: start.toISOString(),
        end_ts: now.toISOString(),
        label,
        confidence: conf_percent() / 100,
        extend_forward: effective_extend,
      });
      set_flash(label);
      set_label_version((v) => v + 1);
      setTimeout(() => set_flash(null), 1500);
    } catch (err: unknown) {
      const pending = overwrite_pending_from_api_error(err, {
        label,
        start: start.toISOString(),
        end: now.toISOString(),
        confidence: conf_percent() / 100,
        extend_forward: effective_extend,
      });
      if (pending) {
        set_overwrite_pending(pending);
        return;
      }
      const msg: string = err instanceof Error ? err.message : "";
      set_error(msg || "Failed to save label");
    }
  }

  async function overwrite_confirm() {
    const pending = overwrite_pending();
    if (!pending) {
      return;
    }
    set_overwrite_pending(null);
    set_error(null);
    try {
      await label_create({
        start_ts: pending.start,
        end_ts: pending.end,
        label: pending.label,
        confidence: pending.confidence,
        extend_forward: pending.extend_forward,
        overwrite: true,
      });
      set_flash(pending.label);
      set_label_version((v) => v + 1);
      setTimeout(() => set_flash(null), 1500);
    } catch (err: unknown) {
      set_error(err instanceof Error ? err.message : "overwrite failed");
    }
  }

  async function keep_all_confirm() {
    const pending = overwrite_pending();
    if (!pending) {
      return;
    }
    set_overwrite_pending(null);
    set_error(null);
    try {
      await label_create({
        start_ts: pending.start,
        end_ts: pending.end,
        label: pending.label,
        confidence: pending.confidence,
        extend_forward: pending.extend_forward,
        allow_overlap: true,
      });
      set_flash(pending.label);
      set_label_version((v) => v + 1);
      setTimeout(() => set_flash(null), 1500);
    } catch (err: unknown) {
      set_error(err instanceof Error ? err.message : "keep all failed");
    }
  }

  async function current_label_stop() {
    const current = current_label();
    if (!current || stop_current_busy()) {
      return;
    }
    const start_ms = iso_date_parse(current.start_ts).getTime();
    const now_ms = Date.now();
    const stop_ts = new Date(now_ms <= start_ms ? start_ms + 1 : now_ms).toISOString();

    set_stop_current_busy(true);
    set_error(null);
    set_flash(null);
    try {
      await label_update({
        start_ts: current.start_ts,
        end_ts: current.end_ts,
        label: current.label,
        new_end_ts: stop_ts,
        extend_forward: false,
      });
      set_label_version((v) => v + 1);
      set_stop_current_pending(false);
    } catch (err: unknown) {
      set_error(err instanceof Error ? err.message : "Failed to stop current label");
    } finally {
      set_stop_current_busy(false);
    }
  }

  return (
    <div
      style={{
        padding: "8px",
        "border-top": "1px solid var(--border)",
        ...(props.max_height != null
          ? { "max-height": `${props.max_height}px`, "overflow-y": "auto" }
          : {}),
      }}
    >
      <Show when={props.suggestion}>
        <PredictionSuggestion
          suggestion={props.suggestion ?? (() => null)}
          on_saved={() => set_label_version((v) => v + 1)}
          on_dismiss={props.on_suggestion_dismiss}
        />
      </Show>

      <LabelTimePicker
        selected_minutes={selected_minutes}
        set_selected_minutes={set_selected_minutes}
        fill_from_last={fill_from_last}
        set_fill_from_last={set_fill_from_last}
        last_label={last_label}
      />

      <ActivitySummary minutes={selected_minutes} prediction={props.prediction} />

      <Show when={selected_minutes() !== 0 || fill_from_last()}>
        <LabelExtendToggle checked={extend_fwd} on_toggle={extend_fwd_toggle} />
      </Show>

      <LabelConfidence value={conf_percent} on_change={set_conf_percent} />

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

      <Show when={flash()}>
        <button
          type="button"
          onClick={() => set_flash(null)}
          style={{
            cursor: "pointer",
            background: "none",
            border: "none",
            padding: "0",
            width: "100%",
            color: "inherit",
            font: "inherit",
          }}
        >
          <LabelFlash flash={flash() ?? ""} />
        </button>
      </Show>

      <Show when={error()}>
        <ErrorBanner message={error() ?? ""} on_close={() => set_error(null)} />
      </Show>

      <div
        style={{
          display: "grid",
          "grid-template-columns": "1fr 1fr",
          gap: "6px",
        }}
      >
        <For each={labels() ?? []}>
          {(lbl) => (
            <button
              type="button"
              onClick={() => label_now(lbl)}
              style={{
                padding: "8px 4px",
                "border-radius": "var(--radius)",
                border: "1px solid var(--border)",
                background: "var(--surface)",
                color: LABEL_COLORS[lbl] ?? "var(--text)",
                cursor: "pointer",
                "font-size": "0.8rem",
                "font-weight": "600",
                "text-align": "center",
                transition: "background 0.1s ease",
              }}
            >
              {lbl}
            </button>
          )}
        </For>
      </div>

      <LabelLast last_label={last_label} />

      <Show when={current_label()}>
        <div
          style={{
            display: "flex",
            "align-items": "center",
            "justify-content": "center",
            gap: "6px",
            "flex-wrap": "wrap",
            "margin-top": "6px",
          }}
        >
          <Show
            when={stop_current_pending()}
            fallback={
              <button
                type="button"
                disabled={stop_current_busy()}
                onClick={() => {
                  set_flash(null);
                  set_error(null);
                  set_stop_current_pending(true);
                }}
                style={{
                  padding: "2px 8px",
                  "border-radius": "6px",
                  border: "1px solid color-mix(in srgb, #ef4444 40%, var(--border))",
                  background: "transparent",
                  color: "#ef4444",
                  cursor: stop_current_busy() ? "not-allowed" : "pointer",
                  "font-size": "0.65rem",
                  opacity: stop_current_busy() ? "0.6" : "0.85",
                }}
              >
                Stop current label
              </button>
            }
          >
            <span style={{ "font-size": "0.62rem", color: "var(--text-muted)" }}>
              Stop recording the current label now?
            </span>
            <button
              type="button"
              disabled={stop_current_busy()}
              onClick={() => set_stop_current_pending(false)}
              style={{
                padding: "2px 8px",
                "border-radius": "6px",
                border: "1px solid var(--border)",
                background: "transparent",
                color: "var(--text-muted)",
                cursor: stop_current_busy() ? "not-allowed" : "pointer",
                "font-size": "0.65rem",
                opacity: stop_current_busy() ? "0.6" : "1",
              }}
            >
              Keep recording
            </button>
            <button
              type="button"
              disabled={stop_current_busy()}
              onClick={() => void current_label_stop()}
              style={{
                padding: "2px 8px",
                "border-radius": "6px",
                border: "1px solid #ef4444",
                background: "#ef4444",
                color: "#fff",
                cursor: stop_current_busy() ? "not-allowed" : "pointer",
                "font-size": "0.65rem",
                "font-weight": "600",
                opacity: stop_current_busy() ? "0.6" : "1",
              }}
            >
              {stop_current_busy() ? "Stopping…" : "Confirm stop"}
            </button>
          </Show>
        </div>
      </Show>
    </div>
  );
};
