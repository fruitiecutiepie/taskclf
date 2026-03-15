import { type Component, Show } from "solid-js";
import { host } from "../lib/host";
import { useWebSocket } from "../lib/ws";
import { LabelRecorder } from "./LabelRecorder";

const CONTENT_W = 280;
const LABEL_MAX_H = 330;
const isBrowserMode = () => window.innerWidth > 300 && !host.isNativeWindow;

export const LabelRecorderWindow: Component = () => {
  const ws = useWebSocket();
  const inBrowser = isBrowserMode();

  function collapse() {
    host.invoke({ cmd: "toggleLabelGrid" });
  }

  return (
    // biome-ignore lint/a11y/noStaticElementInteractions: hover-only container for native window show/hide
    <div
      onMouseEnter={() => {
        if (!inBrowser) {
          host.invoke({ cmd: "cancelLabelHide" });
        }
      }}
      onMouseLeave={() => {
        if (!inBrowser) {
          host.invoke({ cmd: "hideLabelGrid" });
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
          background: "var(--bg)",
          width: inBrowser ? `${CONTENT_W}px` : "100%",
          ...(inBrowser
            ? { "max-height": `${LABEL_MAX_H}px` }
            : { height: "100vh", display: "flex", "flex-direction": "column" }),
          "overflow-y": "auto",
          "border-radius": inBrowser ? "12px" : "0",
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
        <LabelRecorder onCollapse={collapse} prediction={ws.latestPrediction} />
      </div>
    </div>
  );
};
