import { createSignal, createResource, For, Show, type Component } from "solid-js";
import { LiveBadge } from "./components/LiveBadge";
import { StatePanel } from "./components/StatePanel";
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

const MINUTE_OPTIONS = [0, 1, 5, 10, 15, 30, 60] as const;
type TimeUnit = "s" | "m" | "h" | "d";
const UNIT_TO_MINUTES: Record<TimeUnit, number> = { s: 1 / 60, m: 1, h: 60, d: 1440 };

const isPanelView = new URLSearchParams(window.location.search).has("view")
  && new URLSearchParams(window.location.search).get("view") === "panel";

// Dimensions matching window.py (_COMPACT_SIZE, _EXPANDED_SIZE, _PANEL_SIZE)
const WIDGET_W = 260;
const EXPANDED_CONTENT_MAX_H = 268; // 320 (expanded height) - 44 (pill) - 8 (border/gap)
const PANEL_MAX_H = 720;

// In pywebview the viewport width matches the window (260px).
// A normal browser viewport is much wider.
const isBrowserMode = () => window.innerWidth > 300 && !host.isNativeWindow;

const PanelApp: Component = () => {
  const ws = useWebSocket();
  const inBrowser = isBrowserMode();

  return (
    <div
      style={{
        ...(inBrowser
          ? {
              display: "flex",
              "justify-content": "center",
              "padding-top": "32px",
              "min-height": "100vh",
            }
          : {}),
      }}
    >
      <div
        style={{
          background: "transparent",
          width: `${WIDGET_W}px`,
          ...(inBrowser
            ? { "max-height": `${PANEL_MAX_H}px` }
            : { height: "100vh" }),
          overflow: "auto",
          padding: "4px",
        }}
        onMouseEnter={() => host.invoke({ cmd: "cancelPanelHide" })}
        onMouseLeave={() => host.invoke({ cmd: "hideStatePanel" })}
      >
        <StatePanel
          status={ws.connectionStatus}
          latestStatus={ws.latestStatus}
          latestPrediction={ws.latestPrediction}
          latestTrayState={ws.latestTrayState}
          activeSuggestion={ws.activeSuggestion}
          wsStats={ws.wsStats}
        />
      </div>
    </div>
  );
};

const App: Component = () => {
  if (isPanelView) return <PanelApp />;

  const inBrowser = isBrowserMode();
  const [expanded, setExpanded] = createSignal(false);
  const [showPanel, setShowPanel] = createSignal(false);
  let panelHideTimer: number | undefined;

  function browserShowPanel() {
    clearTimeout(panelHideTimer);
    setShowPanel(true);
  }
  function browserScheduleHide() {
    clearTimeout(panelHideTimer);
    panelHideTimer = window.setTimeout(() => setShowPanel(false), 300);
  }
  function browserCancelHide() {
    clearTimeout(panelHideTimer);
  }

  const [labels] = createResource(fetchCoreLabels);
  const [flash, setFlash] = createSignal<string | null>(null);
  const [selectedMinutes, setSelectedMinutes] = createSignal(1);
  const [customActive, setCustomActive] = createSignal(false);
  const [customValue, setCustomValue] = createSignal("");
  const [customUnit, setCustomUnit] = createSignal<TimeUnit>("m");
  const ws = useWebSocket();

  function selectPreset(m: number) {
    setSelectedMinutes(m);
    setCustomActive(false);
    setCustomValue("");
  }

  function applyCustom(raw: string, unit: TimeUnit) {
    const n = parseFloat(raw);
    if (!isNaN(n) && n >= 0) {
      setSelectedMinutes(n * UNIT_TO_MINUTES[unit]);
      setCustomActive(true);
    }
  }

  function expand() {
    setExpanded(true);
    host.invoke({ cmd: "setExpanded" });
  }

  function collapse() {
    setExpanded(false);
    host.invoke({ cmd: "setCompact" });
  }

  async function labelNow(label: string) {
    const mins = selectedMinutes();
    const now = new Date();
    const start = new Date(now.getTime() - mins * 60_000);
    try {
      await createLabel({
        start_ts: start.toISOString().slice(0, -1),
        end_ts: now.toISOString().slice(0, -1),
        label,
        extend_previous: true,
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
        ...(inBrowser
          ? {
              display: "flex",
              "flex-direction": "column",
              "align-items": "center",
              "padding-top": "32px",
              "min-height": "100vh",
            }
          : {}),
      }}
    >
      <div
        style={{
          background: "var(--bg)",
          width: `${WIDGET_W}px`,
          ...(inBrowser
            ? { "box-shadow": "0 4px 24px rgba(0, 0, 0, 0.5)" }
            : { height: "100vh" }),
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
            status={ws.connectionStatus}
            latestStatus={ws.latestStatus}
            latestPrediction={ws.latestPrediction}
            latestTrayState={ws.latestTrayState}
            activeSuggestion={ws.activeSuggestion}
            wsStats={ws.wsStats}
            compact
            onShowPanel={inBrowser ? browserShowPanel : undefined}
            onHidePanel={inBrowser ? browserScheduleHide : undefined}
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

        {/* Expanded: state + label grid */}
        <Show when={expanded()}>
          <div
            style={{
              padding: "8px 12px 12px",
              "border-top": "1px solid var(--border)",
              "max-height": `${EXPANDED_CONTENT_MAX_H}px`,
              "overflow-y": "auto",
            }}
          >
            <div
              style={{
                display: "flex",
                "align-items": "center",
                "justify-content": "center",
                gap: "4px",
                margin: "8px 0",
                "flex-wrap": "wrap",
              }}
            >
              <span
                style={{
                  "font-size": "0.7rem",
                  color: "var(--text-muted)",
                  "margin-right": "2px",
                }}
              >
                Last
              </span>
              <For each={[...MINUTE_OPTIONS]}>
                {(m) => (
                  <button
                    onClick={() => selectPreset(m)}
                    style={{
                      padding: "2px 7px",
                      "border-radius": "10px",
                      border: "1px solid var(--border)",
                      background:
                        !customActive() && selectedMinutes() === m
                          ? "var(--text-muted)"
                          : "var(--surface)",
                      color:
                        !customActive() && selectedMinutes() === m
                          ? "var(--bg)"
                          : "var(--text-muted)",
                      cursor: "pointer",
                      "font-size": "0.7rem",
                      "font-weight":
                        !customActive() && selectedMinutes() === m ? "700" : "500",
                      "line-height": "1.4",
                      transition: "all 0.1s ease",
                    }}
                  >
                    {m === 0 ? "now" : `${m}m`}
                  </button>
                )}
              </For>
              <div
                style={{
                  display: "flex",
                  "align-items": "center",
                  gap: "2px",
                  "margin-left": "2px",
                }}
              >
                <input
                  type="number"
                  min="0"
                  step="any"
                  placeholder="#"
                  value={customValue()}
                  onInput={(e) => {
                    const v = e.currentTarget.value;
                    setCustomValue(v);
                    applyCustom(v, customUnit());
                  }}
                  onFocus={() => {
                    if (customValue()) applyCustom(customValue(), customUnit());
                  }}
                  style={{
                    width: "36px",
                    padding: "2px 4px",
                    "border-radius": "6px",
                    border: `1px solid ${customActive() ? "var(--text-muted)" : "var(--border)"}`,
                    background: "var(--surface)",
                    color: "var(--text-muted)",
                    "font-size": "0.7rem",
                    "text-align": "center",
                    outline: "none",
                  }}
                />
                <select
                  value={customUnit()}
                  onChange={(e) => {
                    const u = e.currentTarget.value as TimeUnit;
                    setCustomUnit(u);
                    if (customValue()) applyCustom(customValue(), u);
                  }}
                  style={{
                    padding: "2px 2px",
                    "border-radius": "6px",
                    border: `1px solid ${customActive() ? "var(--text-muted)" : "var(--border)"}`,
                    background: "var(--surface)",
                    color: "var(--text-muted)",
                    "font-size": "0.7rem",
                    cursor: "pointer",
                    outline: "none",
                  }}
                >
                  <option value="s">s</option>
                  <option value="m">m</option>
                  <option value="h">h</option>
                  <option value="d">d</option>
                </select>
              </div>
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

      {/* Inline state panel for browser mode (hover-to-peek, 300ms dismiss) */}
      <Show when={inBrowser && showPanel()}>
        <div
          style={{
            width: `${WIDGET_W}px`,
            "max-height": `${PANEL_MAX_H}px`,
            "overflow-y": "auto",
            "margin-top": "4px",
          }}
          onMouseEnter={browserCancelHide}
          onMouseLeave={browserScheduleHide}
        >
          <StatePanel
            status={ws.connectionStatus}
            latestStatus={ws.latestStatus}
            latestPrediction={ws.latestPrediction}
            latestTrayState={ws.latestTrayState}
            activeSuggestion={ws.activeSuggestion}
            wsStats={ws.wsStats}
          />
        </div>
      </Show>
    </div>
  );
};

export default App;
