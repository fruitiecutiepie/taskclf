import {
  type Accessor,
  type Component,
  createEffect,
  createSignal,
  on,
  Show,
} from "solid-js";
import { notification_accept, notification_skip } from "../lib/api";
import { time_format } from "../lib/format";
import { LABEL_COLORS } from "../lib/labelColors";
import { frontend_log_error } from "../lib/log";
import { overwrite_pending_from_api_error } from "../lib/overwrite_pending_from_api_error";
import type { LabelSuggestion } from "../lib/ws";
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

function suggestion_range_part_format(d: Date): string {
  return d.toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function suggestion_range_format(
  block_start: string | null | undefined,
  block_end: string | null | undefined,
): string {
  if (!block_start || !block_end) {
    return `${time_format(block_start)} → ${time_format(block_end)}`;
  }

  const start = new Date(block_start);
  const end = new Date(block_end);
  if (Number.isNaN(start.getTime()) || Number.isNaN(end.getTime())) {
    return `${time_format(block_start)} → ${time_format(block_end)}`;
  }

  const crosses_local_day =
    start.getFullYear() !== end.getFullYear()
    || start.getMonth() !== end.getMonth()
    || start.getDate() !== end.getDate();

  if (!crosses_local_day) {
    return `${time_format(block_start)} → ${time_format(block_end)}`;
  }

  return `${suggestion_range_part_format(start)} → ${suggestion_range_part_format(end)}`;
}

export const PredictionSuggestion: Component<{
  suggestion: Accessor<LabelSuggestion | null>;
  on_saved?: () => void;
  on_dismiss?: () => void;
}> = (props) => {
  const s = () => props.suggestion();
  const [error, set_error] = createSignal<string | null>(null);
  const [busy, set_busy] = createSignal(false);
  const [confirm_pending, set_confirm_pending] = createSignal(false);
  const [overwrite_pending, set_overwrite_pending] =
    createSignal<OverwritePending | null>(null);

  createEffect(
    on(
      () => {
        const sg = s();
        return sg ? `${sg.block_start}:${sg.block_end}:${sg.suggested}` : "";
      },
      () => {
        set_overwrite_pending(null);
      },
    ),
  );

  async function suggestion_accept_confirmed() {
    const sg = s();
    if (!sg || busy()) {
      return;
    }
    set_busy(true);
    set_error(null);
    try {
      await notification_accept({
        block_start: sg.block_start,
        block_end: sg.block_end,
        label: sg.suggested,
      });
      set_confirm_pending(false);
      set_overwrite_pending(null);
      props.on_saved?.();
      props.on_dismiss?.();
    } catch (err: unknown) {
      const pending = overwrite_pending_from_api_error(err, {
        label: sg.suggested,
        start: sg.block_start,
        end: sg.block_end,
        confidence: sg.confidence ?? 1,
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

  async function overwrite_confirm() {
    const pending = overwrite_pending();
    if (!pending || busy()) {
      return;
    }
    set_busy(true);
    set_error(null);
    try {
      await notification_accept({
        block_start: pending.start,
        block_end: pending.end,
        label: pending.label,
        overwrite: true,
      });
      set_overwrite_pending(null);
      set_confirm_pending(false);
      props.on_saved?.();
      props.on_dismiss?.();
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
      await notification_accept({
        block_start: pending.start,
        block_end: pending.end,
        label: pending.label,
        allow_overlap: true,
      });
      set_overwrite_pending(null);
      set_confirm_pending(false);
      props.on_saved?.();
      props.on_dismiss?.();
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
    set_busy(true);
    set_error(null);
    try {
      await notification_skip();
      set_confirm_pending(false);
      set_overwrite_pending(null);
      props.on_dismiss?.();
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Failed to dismiss suggestion";
      frontend_log_error("Failed to dismiss suggestion", err);
      set_error(msg);
    } finally {
      set_busy(false);
    }
  }

  const busy_opacity = () => (busy() ? "0.6" : "1");
  const busy_cursor = () => (busy() ? "not-allowed" : "pointer");

  const suggestion_time_range = () => {
    const sg = s();
    return sg ? { start: sg.block_start, end: sg.block_end } : null;
  };

  return (
    <Show when={s()}>
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
                  color: LABEL_COLORS[s()?.old_label ?? ""] ?? "var(--text)",
                  "font-weight": "700",
                }}
              >
                {s()?.old_label}
              </span>
              <span style={{ color: "var(--text-muted)" }}> → </span>
              <span
                style={{
                  color: LABEL_COLORS[s()?.suggested ?? ""] ?? "var(--text)",
                  "font-weight": "700",
                }}
              >
                {s()?.suggested}
              </span>
              <span style={{ color: "var(--text-muted)" }}>
                {" "}
                ({Math.round((s()?.confidence ?? 0) * 100)}%)
              </span>
            </div>
            <div
              style={{
                color: "var(--text-muted)",
                "font-size": "0.7rem",
                "margin-top": "4px",
              }}
            >
              {suggestion_range_format(s()?.block_start, s()?.block_end)}
            </div>
            <Show when={confirm_pending()}>
              <div
                style={{
                  color: "var(--text-muted)",
                  "font-size": "0.65rem",
                  "margin-top": "4px",
                }}
              >
                This will save the suggested label for the range above.
              </div>
            </Show>
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
            <Show
              when={confirm_pending()}
              fallback={
                <>
                  <button
                    type="button"
                    disabled={busy()}
                    onClick={() => set_confirm_pending(true)}
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
                </>
              }
            >
              <button
                type="button"
                disabled={busy()}
                onClick={suggestion_accept_confirmed}
                style={{
                  ...btn_base,
                  border: "1px solid var(--accent, #6366f1)",
                  background: "var(--accent, #6366f1)",
                  color: "#fff",
                  opacity: busy_opacity(),
                  cursor: busy_cursor(),
                }}
              >
                Save label
              </button>
              <button
                type="button"
                disabled={busy()}
                onClick={() => {
                  set_confirm_pending(false);
                  set_overwrite_pending(null);
                }}
                style={{
                  ...btn_base,
                  border: "1px solid var(--border)",
                  background: "var(--surface)",
                  color: "var(--text-muted)",
                  opacity: busy_opacity(),
                  cursor: busy_cursor(),
                }}
              >
                Back
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
            </Show>
          </div>
        </div>
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
          <ErrorBanner message={error() ?? ""} on_close={() => set_error(null)} />
        </Show>
      </div>
    </Show>
  );
};
