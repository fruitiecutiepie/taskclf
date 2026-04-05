import type { Component } from "solid-js";
import { host } from "../lib/host";
import { ws_store_new } from "../lib/ws";
import { HostWindowDragStrip } from "./HostWindowDragStrip";
import { StatusPanel } from "./StatusPanel";

export const StatusPanelWindow: Component = () => {
  const ws = ws_store_new();

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
