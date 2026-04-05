import { type Accessor, type Component, createSignal, Show } from "solid-js";
import { notification_accept, notification_skip } from "../lib/api";
import { time_format } from "../lib/format";
import { frontend_log_error } from "../lib/log";
import type { LabelSuggestion } from "../lib/ws";

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
      setTimeout(() => set_error(null), 4000);
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
      setTimeout(() => set_error(null), 4000);
    } finally {
      set_busy(false);
    }
  }

  return (
    <Show when={s()}>
      <div
        style={{
          background: "var(--surface)",
          border: "1px solid var(--warning)",
          "border-radius": "var(--radius)",
          padding: "12px 16px",
          "margin-bottom": "16px",
          display: "flex",
          "flex-wrap": "wrap",
          "align-items": "center",
          "justify-content": "space-between",
          gap: "12px",
        }}
      >
        <div>
          <div>
            <strong style={{ color: "var(--warning)" }}>Task changed?</strong>{" "}
            <span style={{ color: "var(--text-muted)" }}>
              {s()?.old_label} → suggested:{" "}
            </span>
            <strong>{s()?.suggested}</strong>{" "}
            <span style={{ color: "var(--text-muted)" }}>
              ({Math.round((s()?.confidence ?? 0) * 100)}%)
            </span>
          </div>
          <div
            style={{
              color: "var(--text-muted)",
              "font-size": "0.8rem",
              "margin-top": "4px",
            }}
          >
            Applies to {suggestion_range_format(s()?.block_start, s()?.block_end)}
          </div>
        </div>
        <div style={{ display: "flex", gap: "8px", "flex-shrink": "0" }}>
          <Show
            when={confirm_pending()}
            fallback={
              <button
                type="button"
                disabled={busy()}
                onClick={() => set_confirm_pending(true)}
                style={{
                  padding: "6px 16px",
                  background: "var(--success)",
                  color: "#fff",
                  border: "none",
                  "border-radius": "var(--radius)",
                  cursor: busy() ? "not-allowed" : "pointer",
                  "font-size": "0.85rem",
                  "font-weight": "500",
                  opacity: busy() ? "0.6" : "1",
                }}
              >
                Save Suggested Label
              </button>
            }
          >
            <button
              type="button"
              disabled={busy()}
              onClick={suggestion_accept_confirmed}
              style={{
                padding: "6px 16px",
                background: "var(--warning)",
                color: "#111",
                border: "none",
                "border-radius": "var(--radius)",
                cursor: busy() ? "not-allowed" : "pointer",
                "font-size": "0.85rem",
                "font-weight": "700",
                opacity: busy() ? "0.6" : "1",
              }}
            >
              Confirm Save
            </button>
            <button
              type="button"
              disabled={busy()}
              onClick={() => set_confirm_pending(false)}
              style={{
                padding: "6px 16px",
                background: "var(--border)",
                color: "var(--text-muted)",
                border: "none",
                "border-radius": "var(--radius)",
                cursor: busy() ? "not-allowed" : "pointer",
                "font-size": "0.85rem",
                opacity: busy() ? "0.6" : "1",
              }}
            >
              Cancel
            </button>
          </Show>
          <button
            type="button"
            disabled={busy()}
            onClick={suggestion_dismiss}
            style={{
              padding: "6px 16px",
              background: "var(--border)",
              color: "var(--text-muted)",
              border: "none",
              "border-radius": "var(--radius)",
              cursor: busy() ? "not-allowed" : "pointer",
              "font-size": "0.85rem",
              opacity: busy() ? "0.6" : "1",
            }}
          >
            Dismiss
          </button>
        </div>
        <Show when={error()}>
          <div
            style={{
              color: "var(--error, #e53935)",
              "font-size": "0.8rem",
              "margin-top": "6px",
              width: "100%",
            }}
          >
            Error: {error()}
          </div>
        </Show>
      </div>
    </Show>
  );
};
