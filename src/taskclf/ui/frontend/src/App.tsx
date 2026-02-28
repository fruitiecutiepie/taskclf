import { createSignal, Show, type Component } from "solid-js";
import { LabelGrid } from "./components/LabelGrid";
import { LabelHistory } from "./components/LabelHistory";
import { LiveBadge } from "./components/LiveBadge";
import { StatePanel } from "./components/StatePanel";
import { useWebSocket } from "./lib/ws";
import { host } from "./lib/host";

const isPanelView = new URLSearchParams(window.location.search).has("view")
  && new URLSearchParams(window.location.search).get("view") === "panel";

// Dimensions matching window.py (_COMPACT_SIZE, _EXPANDED_SIZE, _PANEL_SIZE)
const WIDGET_W = 280;
const EXPANDED_CONTENT_MAX_H = 268; // 320 (expanded height) - 44 (pill) - 8 (border/gap)
const PANEL_MAX_H = 750;

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
              background: "url('/bliss.png') center/cover no-repeat fixed",
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
  const [hovering, setHovering] = createSignal(false);
  const [expanded, setExpanded] = createSignal(false);
  const [showPanel, setShowPanel] = createSignal(false);
  const isOpen = () => hovering() || expanded();
  let labelHideTimer: number | undefined;

  function browserTogglePanel() {
    setShowPanel((v) => !v);
  }

  const ws = useWebSocket();

  function showLabel() {
    clearTimeout(labelHideTimer);
    if (!expanded()) {
      setHovering(true);
      host.invoke({ cmd: "setExpanded" });
    }
  }

  function scheduleLabelHide() {
    clearTimeout(labelHideTimer);
    if (!expanded()) {
      labelHideTimer = window.setTimeout(() => {
        setHovering(false);
        host.invoke({ cmd: "setCompact" });
      }, 300);
    }
  }

  function cancelLabelHide() {
    clearTimeout(labelHideTimer);
  }

  function expand() {
    clearTimeout(labelHideTimer);
    setHovering(false);
    setExpanded(true);
    host.invoke({ cmd: "setExpanded" });
  }

  function collapse() {
    setExpanded(false);
    setHovering(false);
    host.invoke({ cmd: "setCompact" });
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
        style={{
          background: "rgba(15, 17, 23, 0.5)",
          "backdrop-filter": "blur(20px)",
          "-webkit-backdrop-filter": "blur(20px)",
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
            compact={!isOpen()}
            onTogglePanel={inBrowser ? browserTogglePanel : undefined}
            onShowLabel={showLabel}
            onHideLabel={scheduleLabelHide}
          />
          <button
            onClick={(e) => {
              e.stopPropagation();
              expanded() ? collapse() : expand();
            }}
            style={{
              background: "none",
              border: "none",
              color: "var(--text)",
              cursor: "pointer",
              "font-size": "0.9rem",
              padding: "4px 8px",
              "line-height": "1",
              transform: expanded() ? "rotate(180deg)" : "none",
              transition: "transform 0.15s ease",
            }}
            title={expanded() ? "Collapse" : "Expand"}
          >
            &#9660;
          </button>
        </div>

        <Show when={hovering() && !expanded()}>
          <div
            style={{ background: "var(--bg)" }}
            onMouseEnter={cancelLabelHide}
            onMouseLeave={scheduleLabelHide}
          >
            <LabelGrid maxHeight={EXPANDED_CONTENT_MAX_H} onCollapse={collapse} prediction={ws.latestPrediction} />
          </div>
        </Show>

        <Show when={expanded()}>
          <div
            style={{
              background: "var(--bg)",
              "max-height": `${EXPANDED_CONTENT_MAX_H}px`,
              "overflow-y": "auto",
            }}
          >
            <LabelHistory visible={expanded} />
          </div>
        </Show>
      </div>

      {/* Inline state panel for browser mode (click to toggle) */}
      <Show when={inBrowser && showPanel()}>
        <div
          style={{
            width: `${WIDGET_W}px`,
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
