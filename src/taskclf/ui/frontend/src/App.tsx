import { createSignal, Show, type Component } from "solid-js";
import { LiveBadge } from "./components/LiveBadge";
import { SuggestionBanner } from "./components/SuggestionBanner";
import { QueuePanel } from "./components/QueuePanel";
import { LabelForm } from "./components/LabelForm";
import { LabelRecent } from "./components/LabelRecent";
import { History } from "./components/History";
import { useWebSocket } from "./lib/ws";
import { host } from "./lib/host";

const tabs = ["Label", "Recent", "Queue", "History"] as const;
type Tab = (typeof tabs)[number];

const App: Component = () => {
  const [activeTab, setActiveTab] = createSignal<Tab>("Label");
  const [expanded, setExpanded] = createSignal(!host.isNativeWindow);
  const ws = useWebSocket();

  async function expand() {
    await host.invoke({ cmd: "setExpanded" });
    setExpanded(true);
  }

  async function collapse() {
    await host.invoke({ cmd: "setCompact" });
    setExpanded(false);
  }

  async function hideWindow() {
    await host.invoke({ cmd: "hideWindow" });
  }

  return (
    <Show
      when={expanded()}
      fallback={
        <div
          onClick={expand}
          class="pywebview-drag-region"
          style={{
            display: "flex",
            "align-items": "center",
            "justify-content": "center",
            height: "100vh",
            padding: "0 16px",
            cursor: "pointer",
            "user-select": "none",
            background: "var(--bg)",
          }}
        >
          <LiveBadge
            prediction={ws.latestPrediction}
            status={ws.connectionStatus}
            compact
          />
        </div>
      }
    >
      <div style={{ "max-width": "960px", margin: "0 auto", padding: "24px" }}>
        <header
          style={{
            display: "flex",
            "align-items": "center",
            "justify-content": "space-between",
            "margin-bottom": "24px",
          }}
        >
          <h1
            style={{
              "font-size": "1.5rem",
              "font-weight": "600",
              color: "var(--text)",
            }}
          >
            taskclf
          </h1>
          <div style={{ display: "flex", "align-items": "center", gap: "8px" }}>
            <LiveBadge
              prediction={ws.latestPrediction}
              status={ws.connectionStatus}
            />
            <Show when={host.isNativeWindow}>
              <button
                onClick={collapse}
                title="Collapse to pill"
                style={{
                  background: "none",
                  border: "none",
                  color: "var(--text-muted)",
                  cursor: "pointer",
                  "font-size": "1.1rem",
                  padding: "2px 6px",
                  "line-height": "1",
                }}
              >
                &minus;
              </button>
              <button
                onClick={hideWindow}
                title="Hide window"
                style={{
                  background: "none",
                  border: "none",
                  color: "var(--text-muted)",
                  cursor: "pointer",
                  "font-size": "1.1rem",
                  padding: "2px 6px",
                  "line-height": "1",
                }}
              >
                &times;
              </button>
            </Show>
          </div>
        </header>

        <SuggestionBanner suggestion={ws.activeSuggestion} />

        <nav
          style={{
            display: "flex",
            gap: "4px",
            "margin-bottom": "24px",
            "border-bottom": "1px solid var(--border)",
            "padding-bottom": "0",
          }}
        >
          {tabs.map((tab) => (
            <button
              onClick={() => setActiveTab(tab)}
              style={{
                padding: "8px 20px",
                border: "none",
                background:
                  activeTab() === tab ? "var(--surface)" : "transparent",
                color:
                  activeTab() === tab ? "var(--accent)" : "var(--text-muted)",
                cursor: "pointer",
                "font-size": "0.9rem",
                "font-weight": activeTab() === tab ? "600" : "400",
                "border-bottom":
                  activeTab() === tab
                    ? "2px solid var(--accent)"
                    : "2px solid transparent",
                "border-radius": "var(--radius) var(--radius) 0 0",
                transition: "all 0.15s ease",
              }}
            >
              {tab}
            </button>
          ))}
        </nav>

        <main>
          {activeTab() === "Label" && <LabelForm />}
          {activeTab() === "Recent" && <LabelRecent />}
          {activeTab() === "Queue" && <QueuePanel />}
          {activeTab() === "History" && <History />}
        </main>
      </div>
    </Show>
  );
};

export default App;
