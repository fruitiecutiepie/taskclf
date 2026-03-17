import { type Component, Show } from "solid-js";
import { host } from "../lib/host";
import { ws_store_new } from "../lib/ws";
import { StatusPanel } from "./StatusPanel";

const CONTENT_W = 280;
const PANEL_MAX_H = 520;
const isBrowserMode = () => window.innerWidth > 300 && !host.isNativeWindow;

export const StatusPanelWindow: Component = () => {
  const ws = ws_store_new();
  const inBrowser = isBrowserMode();

  return (
    <>
      {/* biome-ignore lint/a11y/noStaticElementInteractions: hover-only panel show/hide, not a user interaction */}
      <div
        onMouseEnter={() => {
          if (!inBrowser) {
            host.invoke({ cmd: "cancelPanelHide" });
          }
        }}
        onMouseLeave={() => {
          if (!inBrowser) {
            host.invoke({ cmd: "hideStatePanel" });
          }
        }}
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
          <StatusPanel
            status={ws.connectionStatus}
            latestStatus={ws.latestStatus}
            latestPrediction={ws.latestPrediction}
            latestTrayState={ws.latestTrayState}
            activeSuggestion={ws.activeSuggestion}
            wsStats={ws.wsStats}
            trainState={ws.trainState}
          />
        </div>
      </div>
    </>
  );
};
