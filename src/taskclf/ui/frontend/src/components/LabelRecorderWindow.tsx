import type { Component } from "solid-js";
import { host } from "../lib/host";
import { transition_prompt_notifications_bind } from "../lib/transitionPromptNotifications";
import { ws_store_new } from "../lib/ws";
import { HostWindowDragStrip } from "./HostWindowDragStrip";
import { LabelRecorder } from "./LabelRecorder";

export const LabelRecorderWindow: Component = () => {
  const ws = ws_store_new();
  const label_change_count = ws.label_change_count ?? (() => 0);

  function window_collapse() {
    host.invoke({ cmd: "toggleLabelGrid" });
  }

  transition_prompt_notifications_bind(ws.latest_prompt, () => {
    host.invoke({ cmd: "showLabelGrid" });
  });

  return (
    // biome-ignore lint/a11y/noStaticElementInteractions: hover-only container for native window show/hide
    <div
      onMouseEnter={() => {
        host.invoke({ cmd: "cancelLabelHide" });
      }}
      onMouseLeave={() => {
        host.invoke({ cmd: "hideLabelGrid" });
      }}
    >
      <div
        style={{
          background: "var(--bg)",
          width: "100%",
          height: "100vh",
          display: "flex",
          "flex-direction": "column",
          "overflow-y": "auto",
        }}
      >
        <HostWindowDragStrip />
        <LabelRecorder
          on_collapse={window_collapse}
          prediction={ws.latest_prediction}
          suggestion={ws.active_suggestion}
          suggestions={ws.pending_suggestions}
          label_change_count={label_change_count}
          on_suggestion_dismiss={ws.suggestion_dismiss}
          on_suggestion_select={ws.suggestion_select}
        />
      </div>
    </div>
  );
};
