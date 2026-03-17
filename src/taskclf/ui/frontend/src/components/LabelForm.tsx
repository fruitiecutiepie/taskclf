import { type Component, createResource, createSignal, Show } from "solid-js";
import { core_labels_list, label_create } from "../lib/api";

const input_style = {
  padding: "8px 12px",
  background: "var(--surface)",
  border: "1px solid var(--border)",
  "border-radius": "var(--radius)",
  color: "var(--text)",
  "font-size": "0.9rem",
  width: "100%",
};

const btn_style = {
  padding: "10px 24px",
  background: "var(--accent)",
  color: "#fff",
  border: "none",
  "border-radius": "var(--radius)",
  cursor: "pointer",
  "font-size": "0.9rem",
  "font-weight": "600",
};

export const LabelForm: Component = () => {
  const [labels] = createResource(core_labels_list);
  const [start_ts, set_start_ts] = createSignal("");
  const [end_ts, set_end_ts] = createSignal("");
  const [label, set_label] = createSignal("");
  const [confidence, set_confidence] = createSignal(0.8);
  const [status, set_status] = createSignal<{
    type: "success" | "error";
    msg: string;
  } | null>(null);

  async function label_submit(e: Event) {
    e.preventDefault();
    set_status(null);
    try {
      const result = await label_create({
        start_ts: start_ts(),
        end_ts: end_ts(),
        label: label(),
        confidence: confidence(),
      });
      set_status({
        type: "success",
        msg: `Saved: ${result.label} [${result.start_ts} → ${result.end_ts}]`,
      });
    } catch (err: unknown) {
      set_status({
        type: "error",
        msg: err instanceof Error ? err.message : "Failed to save label",
      });
    }
  }

  return (
    <div>
      <h2
        style={{
          "font-size": "1.15rem",
          "font-weight": "600",
          "margin-bottom": "16px",
        }}
      >
        Add Label Block
      </h2>

      <form
        onSubmit={label_submit}
        style={{
          display: "flex",
          "flex-direction": "column",
          gap: "12px",
          background: "var(--surface)",
          padding: "20px",
          "border-radius": "var(--radius)",
          border: "1px solid var(--border)",
        }}
      >
        <div
          style={{ display: "grid", "grid-template-columns": "1fr 1fr", gap: "12px" }}
        >
          <div>
            <label
              for="lf-start"
              style={{
                "font-size": "0.85rem",
                color: "var(--text-muted)",
                "margin-bottom": "4px",
                display: "block",
              }}
            >
              Start
            </label>
            <input
              id="lf-start"
              type="datetime-local"
              value={start_ts()}
              onInput={(e) => set_start_ts(e.currentTarget.value)}
              required
              style={input_style}
            />
          </div>
          <div>
            <label
              for="lf-end"
              style={{
                "font-size": "0.85rem",
                color: "var(--text-muted)",
                "margin-bottom": "4px",
                display: "block",
              }}
            >
              End
            </label>
            <input
              id="lf-end"
              type="datetime-local"
              value={end_ts()}
              onInput={(e) => set_end_ts(e.currentTarget.value)}
              required
              style={input_style}
            />
          </div>
        </div>

        <div>
          <label
            for="lf-label"
            style={{
              "font-size": "0.85rem",
              color: "var(--text-muted)",
              "margin-bottom": "4px",
              display: "block",
            }}
          >
            Label
          </label>
          <select
            id="lf-label"
            value={label()}
            onChange={(e) => set_label(e.currentTarget.value)}
            required
            style={input_style}
          >
            <option value="">Select a label...</option>
            {(labels() ?? []).map((l) => (
              <option value={l}>{l}</option>
            ))}
          </select>
        </div>

        <div>
          <label
            for="lf-confidence"
            style={{
              "font-size": "0.85rem",
              color: "var(--text-muted)",
              "margin-bottom": "4px",
              display: "block",
            }}
          >
            Confidence: {confidence().toFixed(2)}
          </label>
          <input
            id="lf-confidence"
            type="range"
            min="0"
            max="1"
            step="0.05"
            value={confidence()}
            onInput={(e) => set_confidence(parseFloat(e.currentTarget.value))}
            style={{ width: "100%", "margin-top": "4px" }}
          />
        </div>

        <button type="submit" style={btn_style}>
          Save Label Block
        </button>
      </form>

      <Show when={status()}>
        <div
          style={{
            "margin-top": "12px",
            padding: "10px 16px",
            "border-radius": "var(--radius)",
            background:
              status()?.type === "success"
                ? "rgba(34,197,94,0.15)"
                : "rgba(239,68,68,0.15)",
            color: status()?.type === "success" ? "var(--success)" : "var(--danger)",
            "font-size": "0.9rem",
          }}
        >
          {status()?.msg}
        </div>
      </Show>
    </div>
  );
};
