import type { Component } from "solid-js";
import { host } from "../lib/host";

/** Top grab bar for label/panel child windows (pywebview CSS drag region or Electron app-region). */
export const HostWindowDragStrip: Component = () => {
  return (
    <div
      class={host.kind === "pywebview" ? "pywebview-drag-region" : undefined}
      style={{
        height: "10px",
        cursor: "grab",
        "flex-shrink": "0",
        display: "flex",
        "justify-content": "center",
        "align-items": "center",
        ...(host.kind === "electron"
          ? { "-webkit-app-region": "drag", "app-region": "drag" }
          : {}),
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
  );
};
