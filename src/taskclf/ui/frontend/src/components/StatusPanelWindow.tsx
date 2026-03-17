import { type Component, Show } from "solid-js";
import { host } from "../lib/host";
import { ws_store_new } from "../lib/ws";
import { StatusPanel } from "./StatusPanel";

const CONTENT_W = 280;
const PANEL_MAX_H = 520;
const is_browser_mode = () => window.innerWidth > 300 && !host.isNativeWindow;

export const StatusPanelWindow: Component = () => {
  const ws = ws_store_new();
  const in_browser = is_browser_mode();

  return (
    <>
      {/* biome-ignore lint/a11y/noStaticElementInteractions: hover-only panel show/hide, not a user interaction */}
      <div
        onMouseEnter={() => {
          if (!in_browser) {
            host.invoke({ cmd: "cancelPanelHide" });
          }
        }}
        onMouseLeave={() => {
          if (!in_browser) {
            host.invoke({ cmd: "hideStatePanel" });
          }
        }}
        style={{
          ...(in_browser
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
            width: in_browser ? `${CONTENT_W}px` : "100%",
            ...(in_browser
              ? { "max-height": `${PANEL_MAX_H}px` }
              : { height: "100vh", display: "flex", "flex-direction": "column" }),
            overflow: "auto",
            padding: "4px",
          }}
        >
          <Show when={!in_browser}>
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
            status={ws.connection_status}
            latest_status={ws.latest_status}
            latest_prediction={ws.latest_prediction}
            latest_tray_state={ws.latest_tray_state}
            active_suggestion={ws.active_suggestion}
            ws_stats={ws.ws_stats}
            train_state={ws.train_state}
          />
        </div>
      </div>
    </>
  );
};
