import {
  type Component,
  createResource,
  createSignal,
  For,
  Show,
} from "solid-js";
import { createLabel, fetchAWLive, fetchCoreLabels } from "../lib/api";

const inputStyle = {
  padding: "8px 12px",
  background: "var(--surface)",
  border: "1px solid var(--border)",
  "border-radius": "var(--radius)",
  color: "var(--text)",
  "font-size": "0.9rem",
  width: "100%",
};

export const LabelRecent: Component = () => {
  const [labels] = createResource(fetchCoreLabels);
  const [minutes, setMinutes] = createSignal(10);
  const [label, setLabel] = createSignal("");
  const [userId, setUserId] = createSignal("default-user");
  const [confidence, setConfidence] = createSignal(0.8);
  const [status, setStatus] = createSignal<{
    type: "success" | "error";
    msg: string;
  } | null>(null);
  const [awApps, setAwApps] = createSignal<
    { app: string; events: number }[] | null
  >(null);

  async function submit(e: Event) {
    e.preventDefault();
    setStatus(null);
    setAwApps(null);

    const now = new Date();
    const start = new Date(now.getTime() - minutes() * 60_000);
    const startIso = start.toISOString().slice(0, -1);
    const endIso = now.toISOString().slice(0, -1);

    try {
      const apps = await fetchAWLive(startIso, endIso);
      if (apps.length) setAwApps(apps);
    } catch {
      // AW unavailable
    }

    try {
      const result = await createLabel({
        start_ts: startIso,
        end_ts: endIso,
        label: label(),
        user_id: userId(),
        confidence: confidence(),
      });
      setStatus({
        type: "success",
        msg: `Saved: ${result.label} [last ${minutes()} min]`,
      });
    } catch (err: any) {
      setStatus({ type: "error", msg: err.message || "Failed" });
    }
  }

  return (
    <div>
      <h2
        style={{
          "font-size": "1.15rem",
          "font-weight": "600",
          "margin-bottom": "8px",
        }}
      >
        Label Recent Activity
      </h2>
      <p
        style={{
          color: "var(--text-muted)",
          "font-size": "0.85rem",
          "margin-bottom": "16px",
        }}
      >
        Quickly label what you've been doing in the last few minutes.
      </p>

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
        <div>
          <label
            style={{
              "font-size": "0.85rem",
              color: "var(--text-muted)",
              "margin-bottom": "4px",
              display: "block",
            }}
          >
            Last {minutes()} minutes
          </label>
          <input
            type="range"
            min="1"
            max="60"
            step="1"
            value={minutes()}
            onInput={(e) => setMinutes(parseInt(e.currentTarget.value))}
            style={{ width: "100%" }}
          />
        </div>

        <div>
          <label
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

        <div
          style={{
            display: "grid",
            "grid-template-columns": "1fr 1fr",
            gap: "12px",
          }}
        >
          <div>
            <label
              style={{
                "font-size": "0.85rem",
                color: "var(--text-muted)",
                "margin-bottom": "4px",
                display: "block",
              }}
            >
              User ID
            </label>
            <input
              type="text"
              value={userId()}
              onInput={(e) => setUserId(e.currentTarget.value)}
              style={inputStyle}
            />
          </div>
          <div>
            <label
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
              type="range"
              min="0"
              max="1"
              step="0.05"
              value={confidence()}
              onInput={(e) =>
                setConfidence(parseFloat(e.currentTarget.value))
              }
              style={{ width: "100%", "margin-top": "4px" }}
            />
          </div>
        </div>

        <button
          type="submit"
          style={{
            padding: "10px 24px",
            background: "var(--accent)",
            color: "#fff",
            border: "none",
            "border-radius": "var(--radius)",
            cursor: "pointer",
            "font-size": "0.9rem",
            "font-weight": "600",
          }}
        >
          Label Now
        </button>
      </form>

      <Show when={awApps()}>
        <div
          style={{
            "margin-top": "16px",
            background: "var(--surface)",
            "border-radius": "var(--radius)",
            border: "1px solid var(--border)",
            padding: "16px",
          }}
        >
          <h3
            style={{
              "font-size": "0.95rem",
              "font-weight": "600",
              "margin-bottom": "8px",
            }}
          >
            ActivityWatch Summary
          </h3>
          <table style={{ width: "100%", "border-collapse": "collapse" }}>
            <thead>
              <tr>
                <th
                  style={{
                    "text-align": "left",
                    padding: "4px 8px",
                    color: "var(--text-muted)",
                    "font-size": "0.8rem",
                    "font-weight": "500",
                  }}
                >
                  App
                </th>
                <th
                  style={{
                    "text-align": "right",
                    padding: "4px 8px",
                    color: "var(--text-muted)",
                    "font-size": "0.8rem",
                    "font-weight": "500",
                  }}
                >
                  Events
                </th>
              </tr>
            </thead>
            <tbody>
              <For each={awApps()!}>
                {(entry) => (
                  <tr>
                    <td
                      style={{
                        padding: "4px 8px",
                        "font-size": "0.85rem",
                      }}
                    >
                      {entry.app}
                    </td>
                    <td
                      style={{
                        padding: "4px 8px",
                        "text-align": "right",
                        "font-size": "0.85rem",
                        color: "var(--text-muted)",
                      }}
                    >
                      {entry.events}
                    </td>
                  </tr>
                )}
              </For>
            </tbody>
          </table>
        </div>
      </Show>

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
