import { createSignal, createResource, For, Show, type Component } from "solid-js";
import { LiveBadge } from "./components/LiveBadge";
import { useWebSocket } from "./lib/ws";
import { host } from "./lib/host";
import { createLabel, fetchCoreLabels } from "./lib/api";

const LABEL_COLORS: Record<string, string> = {
  Build: "#6366f1",
  Debug: "#f59e0b",
  Review: "#8b5cf6",
  Write: "#3b82f6",
  ReadResearch: "#14b8a6",
  Communicate: "#f97316",
  Meet: "#ec4899",
  BreakIdle: "#6b7280",
};

const DEFAULT_LABEL_MINUTES = 5;

const App: Component = () => {
  const [expanded, setExpanded] = createSignal(false);
  const [labels] = createResource(fetchCoreLabels);
  const [flash, setFlash] = createSignal<string | null>(null);
  const ws = useWebSocket();

  function expand() {
    setExpanded(true);
    host.invoke({ cmd: "setExpanded" });
  }

  function collapse() {
    setExpanded(false);
    host.invoke({ cmd: "setCompact" });
  }

  async function labelNow(label: string) {
    const now = new Date();
    const start = new Date(now.getTime() - DEFAULT_LABEL_MINUTES * 60_000);
    try {
      await createLabel({
        start_ts: start.toISOString().slice(0, -1),
        end_ts: now.toISOString().slice(0, -1),
        label,
        user_id: "default-user",
      });
      setFlash(label);
      setTimeout(() => {
        setFlash(null);
        collapse();
      }, 1500);
    } catch (err: any) {
      setFlash(`Error: ${err.message}`);
      setTimeout(() => setFlash(null), 3000);
    }
  }

  return (
    <div
      style={{
        background: "var(--bg)",
        height: "100vh",
        overflow: "hidden",
        "border-radius": "12px",
      }}
    >
      {/* Compact pill */}
      <div
        class="pywebview-drag-region"
        style={{
          display: "flex",
          "align-items": "center",
          "justify-content": "space-between",
          padding: "0 8px 0 12px",
          height: "44px",
          "user-select": "none",
        }}
      >
        <LiveBadge
          prediction={ws.latestPrediction}
          status={ws.connectionStatus}
          compact
        />
        <button
          onClick={(e) => {
            e.stopPropagation();
            expanded() ? collapse() : expand();
          }}
          style={{
            background: "none",
            border: "none",
            color: "var(--text-muted)",
            cursor: "pointer",
            "font-size": "0.9rem",
            padding: "4px 8px",
            "line-height": "1",
            transform: expanded() ? "rotate(180deg)" : "none",
            transition: "transform 0.15s ease",
          }}
          title={expanded() ? "Collapse" : "Label now"}
        >
          &#9660;
        </button>
      </div>

      {/* Expanded: label grid */}
      <Show when={expanded()}>
        <div
          style={{
            padding: "8px 12px 12px",
            "border-top": "1px solid var(--border)",
          }}
        >
          <div
            style={{
              "font-size": "0.75rem",
              color: "var(--text-muted)",
              "margin-bottom": "8px",
              "text-align": "center",
            }}
          >
            Label last {DEFAULT_LABEL_MINUTES} min
          </div>

          <Show when={flash()}>
            <div
              style={{
                "text-align": "center",
                "font-size": "0.8rem",
                "margin-bottom": "8px",
                color: flash()!.startsWith("Error")
                  ? "var(--danger)"
                  : "var(--success)",
              }}
            >
              {flash()!.startsWith("Error") ? flash() : `Saved: ${flash()}`}
            </div>
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
                  onClick={() => labelNow(lbl)}
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
        </div>
      </Show>
    </div>
  );
};

export default App;
