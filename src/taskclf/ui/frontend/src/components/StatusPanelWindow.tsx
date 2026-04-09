import type { Component } from "solid-js";
import { host } from "../lib/host";
import { transition_prompt_notifications_bind } from "../lib/transitionPromptNotifications";
import { ws_store_new } from "../lib/ws";
import { HostWindowDragStrip } from "./HostWindowDragStrip";
import { StatusPanel } from "./StatusPanel";

export const StatusPanelWindow: Component = () => {
  const ws = ws_store_new();
  const label_change_count = ws.label_change_count ?? (() => 0);

  transition_prompt_notifications_bind(ws.latest_prompt, () => {
    host.invoke({ cmd: "showLabelGrid" });
  });

  return (
    <>
      {/* biome-ignore lint/a11y/noStaticElementInteractions: hover-only panel show/hide, not a user interaction */}
      <div
        onMouseEnter={() => {
          host.invoke({ cmd: "cancelPanelHide" });
        }}
        onMouseLeave={() => {
          host.invoke({ cmd: "hideStatePanel" });
        }}
      >
        <div
          style={{
            background: "transparent",
            width: "100%",
            height: "100vh",
            display: "flex",
            "flex-direction": "column",
            overflow: "auto",
            padding: "4px",
          }}
        >
          <HostWindowDragStrip />
          <StatusPanel
            status={ws.connection_status}
            latest_status={ws.latest_status}
            latest_prediction={ws.latest_prediction}
            latest_tray_state={ws.latest_tray_state}
            active_suggestion={ws.active_suggestion}
            label_change_count={label_change_count}
            ws_stats={ws.ws_stats}
            train_state={ws.train_state}
            on_open_label_recorder={() => {
              host.invoke({ cmd: "showLabelGrid" });
            }}
          />
        </div>
      </div>
    </>
  );
};
