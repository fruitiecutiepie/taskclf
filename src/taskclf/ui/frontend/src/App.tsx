import { createSignal, Show, type Component } from "solid-js";
import { LabelGrid } from "./components/LabelGrid";
import { LiveBadge } from "./components/LiveBadge";
import { StatePanel } from "./components/StatePanel";
import { useWebSocket } from "./lib/ws";
import { host } from "./lib/host";

const viewParam = new URLSearchParams(window.location.search).get("view");
const isPanelView = viewParam === "panel";
const isLabelView = viewParam === "label";

// Dimensions matching window.py (_COMPACT_SIZE, _LABEL_SIZE, _PANEL_SIZE)
const COMPACT_W = 150;
const CONTENT_W = 280;
const LABEL_MAX_H = 330;
const PANEL_MAX_H = 520;

const isBrowserMode = () => window.innerWidth > 300 && !host.isNativeWindow;

if (!isBrowserMode()) {
  document.documentElement.style.background = "transparent";
  document.body.style.background = "transparent";
}

/* ---------- Label grid window (standalone pywebview) ---------- */

const LabelApp: Component = () => {
  const ws = useWebSocket();
  const inBrowser = isBrowserMode();

  function collapse() {
    host.invoke({ cmd: "hideLabelGrid" });
  }

  return (
    <div
      style={{
        ...(inBrowser
          ? {
              display: "flex",
              "justify-content": "center",
              "padding-top": "32px",
              "min-height": "100vh",
              background: "url('/bliss.png') center/cover no-repeat fixed",
            }
          : {}),
      }}
    >
      <div
        style={{
          background: "var(--bg)",
          width: inBrowser ? `${CONTENT_W}px` : "100%",
          ...(inBrowser
            ? { "max-height": `${LABEL_MAX_H}px` }
            : { height: "100vh" }),
          "overflow-y": "auto",
          "border-radius": inBrowser ? "12px" : "0",
        }}
      >
        <LabelGrid onCollapse={collapse} prediction={ws.latestPrediction} />
      </div>
    </div>
  );
};

/* ---------- State panel window (standalone pywebview) ---------- */

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
              background: "url('/bliss.png') center/cover no-repeat fixed",
            }
          : {}),
      }}
    >
      <div
        style={{
          background: "transparent",
          width: inBrowser ? `${CONTENT_W}px` : "100%",
          ...(inBrowser
            ? { "max-height": `${PANEL_MAX_H}px` }
            : { height: "100vh", display: "flex", "flex-direction": "column" }),
          overflow: "auto",
          padding: "4px",
        }}
      >
        <Show when={!inBrowser}>
          <div
            class="pywebview-drag-region"
            style={{
              height: "10px",
              cursor: "grab",
              "flex-shrink": "0",
              display: "flex",
              "justify-content": "center",
              "align-items": "center",
            }}
          >
            <div
              style={{
                width: "32px",
                height: "3px",
                "border-radius": "2px",
                background: "rgba(255,255,255,0.15)",
              }}
            />
          </div>
        </Show>
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

/* ---------- Main pill window ---------- */

const App: Component = () => {
  if (isLabelView) return <LabelApp />;
  if (isPanelView) return <PanelApp />;

  const inBrowser = isBrowserMode();
  const [hovering, setHovering] = createSignal(false);
  const [showPanel, setShowPanel] = createSignal(false);

  function browserTogglePanel() {
    setShowPanel((v) => !v);
  }

  const ws = useWebSocket();

  function showLabel() {
    setHovering(true);
    host.invoke({ cmd: "showLabelGrid" });
  }

  function hideLabel() {
    setHovering(false);
    host.invoke({ cmd: "hideLabelGrid" });
  }

  return (
    <div
      style={{
        ...(inBrowser
          ? {
              display: "flex",
              "flex-direction": "column",
              "align-items": "flex-end",
              "padding-top": "16px",
              "padding-right": "16px",
              "min-height": "100vh",
              background: "url('/bliss.png') center/cover no-repeat fixed",
            }
          : {}),
      }}
    >
      <div
        onMouseEnter={inBrowser ? () => setHovering(true) : undefined}
        onMouseLeave={inBrowser ? () => setHovering(false) : undefined}
        style={{
          display: "flex",
          "flex-direction": "column",
          "align-items": "flex-end",
        }}
      >
        <div
          style={{
            background: "rgba(15, 17, 23, 0.5)",
            "backdrop-filter": "blur(20px)",
            "-webkit-backdrop-filter": "blur(20px)",
            width: inBrowser ? `${COMPACT_W}px` : "100%",
            ...(inBrowser
              ? { "box-shadow": "0 4px 24px rgba(0, 0, 0, 0.5)" }
              : { height: "100vh" }),
            overflow: "hidden",
            "border-radius": "20px",
          }}
        >
          <div
            class="pywebview-drag-region"
            style={{
              display: "flex",
              "align-items": "center",
              "justify-content": "center",
              padding: "0 8px 0 12px",
              height: "30px",
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
              compact={!hovering()}
              onTogglePanel={inBrowser ? browserTogglePanel : undefined}
              onShowLabel={inBrowser ? undefined : showLabel}
              onHideLabel={inBrowser ? undefined : hideLabel}
            />
          </div>
        </div>

        {/* Inline label grid for browser mode preview */}
        <Show when={inBrowser && hovering()}>
          <div
            style={{
              width: `${CONTENT_W}px`,
              "max-height": `${LABEL_MAX_H}px`,
              "overflow-y": "auto",
              "margin-top": "4px",
              background: "var(--bg)",
              "border-radius": "12px",
            }}
          >
            <LabelGrid onCollapse={hideLabel} prediction={ws.latestPrediction} />
          </div>
        </Show>
      </div>

      {/* Inline state panel for browser mode preview */}
      <Show when={inBrowser && showPanel()}>
        <div
          style={{
            width: `${CONTENT_W}px`,
            "max-height": `${PANEL_MAX_H}px`,
            "overflow-y": "auto",
            "margin-top": "4px",
          }}
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
