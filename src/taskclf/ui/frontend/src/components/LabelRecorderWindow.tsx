import { type Component, Show } from "solid-js";
import { host } from "../lib/host";
import { ws_store_new } from "../lib/ws";
import { LabelRecorder } from "./LabelRecorder";

const CONTENT_W = 280;
const LABEL_MAX_H = 330;
const is_browser_mode = () => window.innerWidth > 300 && !host.isNativeWindow;

export const LabelRecorderWindow: Component = () => {
  const ws = ws_store_new();
  const in_browser = is_browser_mode();

  function window_collapse() {
    host.invoke({ cmd: "toggleLabelGrid" });
  }

  return (
    // biome-ignore lint/a11y/noStaticElementInteractions: hover-only container for native window show/hide
    <div
      onMouseEnter={() => {
        if (!in_browser) {
          host.invoke({ cmd: "cancelLabelHide" });
        }
      }}
      onMouseLeave={() => {
        if (!in_browser) {
          host.invoke({ cmd: "hideLabelGrid" });
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
          background: "var(--bg)",
          width: in_browser ? `${CONTENT_W}px` : "100%",
          ...(in_browser
            ? { "max-height": `${LABEL_MAX_H}px` }
            : { height: "100vh", display: "flex", "flex-direction": "column" }),
          "overflow-y": "auto",
          "border-radius": in_browser ? "12px" : "0",
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
        <LabelRecorder
          on_collapse={window_collapse}
          prediction={ws.latest_prediction}
          suggestion={ws.active_suggestion}
          on_suggestion_dismiss={ws.suggestion_dismiss}
        />
      </div>
    </div>
  );
};
