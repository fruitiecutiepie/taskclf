import { type Accessor, type Component, Show, createSignal } from "solid-js";
import type { LabelSuggestion } from "../lib/ws";
import { createLabel } from "../lib/api";

export const PredictionSuggestion: Component<{
  suggestion: Accessor<LabelSuggestion | null>;
}> = (props) => {
  const s = () => props.suggestion();
  const [error, setError] = createSignal<string | null>(null);

  async function accept() {
    const sg = s();
    if (!sg) return;
    setError(null);
    try {
      await createLabel({
        start_ts: sg.block_start,
        end_ts: sg.block_end,
        label: sg.suggested,
        confidence: sg.confidence,
      });
    } catch (err: any) {
      const msg = err?.message || "Failed to accept suggestion";
      console.error("Failed to accept suggestion", err);
      setError(msg);
      setTimeout(() => setError(null), 4000);
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
          <strong style={{ color: "var(--warning)" }}>Task changed?</strong>{" "}
          <span style={{ color: "var(--text-muted)" }}>
            {s()!.old_label} → suggested:{" "}
          </span>
          <strong>{s()!.suggested}</strong>{" "}
          <span style={{ color: "var(--text-muted)" }}>
            ({Math.round(s()!.confidence * 100)}%)
          </span>
        </div>
        <div style={{ display: "flex", gap: "8px", "flex-shrink": "0" }}>
          <button
            onClick={accept}
            style={{
              padding: "6px 16px",
              background: "var(--success)",
              color: "#fff",
              border: "none",
              "border-radius": "var(--radius)",
              cursor: "pointer",
              "font-size": "0.85rem",
              "font-weight": "500",
            }}
          >
            Accept
          </button>
          <button
            onClick={() => {
              /* parent dismisses via ws.dismissSuggestion */
            }}
            style={{
              padding: "6px 16px",
              background: "var(--border)",
              color: "var(--text-muted)",
              border: "none",
              "border-radius": "var(--radius)",
              cursor: "pointer",
              "font-size": "0.85rem",
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
