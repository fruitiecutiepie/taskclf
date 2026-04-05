import { type Accessor, type Component, createSignal, Show } from "solid-js";
import { notification_accept, notification_skip } from "../lib/api";
import { time_format } from "../lib/format";
import { LABEL_COLORS } from "../lib/labelColors";
import { frontend_log_error } from "../lib/log";
import type { LabelSuggestion } from "../lib/ws";
import { ErrorBanner } from "./ErrorBanner";

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
      props.on_saved?.();
      props.on_dismiss?.();
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Failed to save inferred label";
      frontend_log_error("Failed to save inferred label", err);
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

  return (
    <Show when={s()}>
      <div
        style={{
          background: "var(--surface)",
          border: "1px solid var(--border)",
          "border-left": "3px solid var(--warning)",
          "border-radius": "var(--radius)",
          padding: "8px 10px",
          "margin-bottom": "8px",
          display: "flex",
          "flex-direction": "column",
          gap: "6px",
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
                onClick={() => set_confirm_pending(false)}
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
        <Show when={error()}>
          <ErrorBanner message={error() ?? ""} on_close={() => set_error(null)} />
        </Show>
      </div>
    </Show>
  );
};
