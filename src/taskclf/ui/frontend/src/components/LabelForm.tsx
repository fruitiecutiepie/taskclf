import {
  type Component,
  createResource,
  createSignal,
  Show,
} from "solid-js";
import { createLabel, fetchCoreLabels } from "../lib/api";

const inputStyle = {
  padding: "8px 12px",
  background: "var(--surface)",
  border: "1px solid var(--border)",
  "border-radius": "var(--radius)",
  color: "var(--text)",
  "font-size": "0.9rem",
  width: "100%",
};

const btnStyle = {
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
  const [labels] = createResource(fetchCoreLabels);
  const [startTs, setStartTs] = createSignal("");
  const [endTs, setEndTs] = createSignal("");
  const [label, setLabel] = createSignal("");
  const [confidence, setConfidence] = createSignal(0.8);
  const [status, setStatus] = createSignal<{
    type: "success" | "error";
    msg: string;
  } | null>(null);

  async function submit(e: Event) {
    e.preventDefault();
    setStatus(null);
    try {
      const result = await createLabel({
        start_ts: startTs(),
        end_ts: endTs(),
        label: label(),
        confidence: confidence(),
      });
      setStatus({
        type: "success",
        msg: `Saved: ${result.label} [${result.start_ts} → ${result.end_ts}]`,
      });
    } catch (err: any) {
      setStatus({ type: "error", msg: err.message || "Failed to save label" });
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
        onSubmit={submit}
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
        <div style={{ display: "grid", "grid-template-columns": "1fr 1fr", gap: "12px" }}>
          <div>
            <label style={{ "font-size": "0.85rem", color: "var(--text-muted)", "margin-bottom": "4px", display: "block" }}>
              Start
            </label>
            <input
              type="datetime-local"
              value={startTs()}
              onInput={(e) => setStartTs(e.currentTarget.value)}
              required
              style={inputStyle}
            />
          </div>
          <div>
            <label style={{ "font-size": "0.85rem", color: "var(--text-muted)", "margin-bottom": "4px", display: "block" }}>
              End
            </label>
            <input
              type="datetime-local"
              value={endTs()}
              onInput={(e) => setEndTs(e.currentTarget.value)}
              required
              style={inputStyle}
            />
          </div>
        </div>

        <div>
          <label style={{ "font-size": "0.85rem", color: "var(--text-muted)", "margin-bottom": "4px", display: "block" }}>
            Label
          </label>
          <select
            value={label()}
            onChange={(e) => setLabel(e.currentTarget.value)}
            required
            style={inputStyle}
          >
            <option value="">Select a label...</option>
            {(labels() ?? []).map((l) => (
              <option value={l}>{l}</option>
            ))}
          </select>
        </div>

        <div>
          <label style={{ "font-size": "0.85rem", color: "var(--text-muted)", "margin-bottom": "4px", display: "block" }}>
            Confidence: {confidence().toFixed(2)}
          </label>
          <input
            type="range"
            min="0"
            max="1"
            step="0.05"
            value={confidence()}
            onInput={(e) => setConfidence(parseFloat(e.currentTarget.value))}
            style={{ width: "100%", "margin-top": "4px" }}
          />
        </div>

        <button type="submit" style={btnStyle}>
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
              status()!.type === "success"
                ? "rgba(34,197,94,0.15)"
                : "rgba(239,68,68,0.15)",
            color:
              status()!.type === "success"
                ? "var(--success)"
                : "var(--danger)",
            "font-size": "0.9rem",
          }}
        >
          {status()!.msg}
        </div>
      </Show>
    </div>
  );
};
