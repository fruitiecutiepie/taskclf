import { createSignal, Show, type Component } from "solid-js";
import { LabelGrid } from "./components/LabelGrid";
import { LabelHistory } from "./components/LabelHistory";
import { LiveBadge } from "./components/LiveBadge";
import { StatePanel } from "./components/StatePanel";
import { useWebSocket } from "./lib/ws";
import { host } from "./lib/host";

const viewParam = new URLSearchParams(window.location.search).get("view");
const isPanelView = viewParam === "panel";
const isHistoryView = viewParam === "history";

// Dimensions matching window.py (_COMPACT_SIZE, _EXPANDED_SIZE, _PANEL_SIZE, _HISTORY_SIZE)
const WIDGET_W = 280;
const EXPANDED_CONTENT_MAX_H = 268; // 320 (expanded height) - 44 (pill) - 8 (border/gap)
const PANEL_MAX_H = 750;
const HISTORY_MAX_H = 400;

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

const HistoryApp: Component = () => {
  const inBrowser = isBrowserMode();
  const visible = () => true;

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
            ? { "max-height": `${HISTORY_MAX_H}px` }
            : { height: "100vh" }),
          overflow: "auto",
          padding: "4px",
        }}
      >
        <LabelHistory visible={visible} />
      </div>
    </div>
  );
};

const App: Component = () => {
  if (isPanelView) return <PanelApp />;
  if (isHistoryView) return <HistoryApp />;

  const inBrowser = isBrowserMode();
  const [hovering, setHovering] = createSignal(false);
  const [historyOpen, setHistoryOpen] = createSignal(false);
  const [showPanel, setShowPanel] = createSignal(false);
  const [showHistory, setShowHistory] = createSignal(false);
  let labelHideTimer: number | undefined;

  function browserTogglePanel() {
    setShowPanel((v) => !v);
  }

  const ws = useWebSocket();

  function showLabel() {
    clearTimeout(labelHideTimer);
    setHovering(true);
    host.invoke({ cmd: "setExpanded" });
  }

  function scheduleLabelHide() {
    clearTimeout(labelHideTimer);
    labelHideTimer = window.setTimeout(() => {
      setHovering(false);
      host.invoke({ cmd: "setCompact" });
    }, 300);
  }

  function cancelLabelHide() {
    clearTimeout(labelHideTimer);
  }

  function collapseLabel() {
    clearTimeout(labelHideTimer);
    setHovering(false);
    host.invoke({ cmd: "setCompact" });
  }

  function toggleHistory() {
    const next = !historyOpen();
    setHistoryOpen(next);
    if (inBrowser) {
      setShowHistory(next);
    } else {
      host.invoke({ cmd: "toggleLabelHistory" });
    }
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
            compact={!hovering()}
            onTogglePanel={inBrowser ? browserTogglePanel : undefined}
            onShowLabel={showLabel}
            onHideLabel={scheduleLabelHide}
          />
          <button
            onClick={(e) => {
              e.stopPropagation();
              toggleHistory();
            }}
            style={{
              background: "none",
              border: "none",
              color: "var(--text)",
              cursor: "pointer",
              "font-size": "0.9rem",
              padding: "4px 8px",
              "line-height": "1",
              transform: historyOpen() ? "rotate(180deg)" : "none",
              transition: "transform 0.15s ease",
            }}
            title={historyOpen() ? "Hide history" : "Show history"}
          >
            &#9660;
          </button>
        </div>

        <Show when={hovering()}>
          <div
            style={{ background: "var(--bg)" }}
            onMouseEnter={cancelLabelHide}
            onMouseLeave={scheduleLabelHide}
          >
            <LabelGrid maxHeight={EXPANDED_CONTENT_MAX_H} onCollapse={collapseLabel} prediction={ws.latestPrediction} />
          </div>
        </Show>
      </div>

      {/* Inline label history for browser mode */}
      <Show when={inBrowser && showHistory()}>
        <div
          style={{
            width: `${WIDGET_W}px`,
            "max-height": `${HISTORY_MAX_H}px`,
            "overflow-y": "auto",
            "margin-top": "4px",
          }}
        >
          <LabelHistory visible={showHistory} />
        </div>
      </Show>

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
