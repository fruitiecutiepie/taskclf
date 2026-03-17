import { type Accessor, type Component, createSignal } from "solid-js";
import { dot_color } from "../lib/labelColors";
import type { ConnectionStatus } from "../lib/ws";

export const ConnectionDot: Component<{
  status: Accessor<ConnectionStatus>;
  panel_pinned?: Accessor<boolean>;
  on_toggle_panel?: () => void;
  on_show_panel?: () => void;
  on_hide_panel?: () => void;
}> = (props) => {
  const [hovered, set_hovered] = createSignal(false);
  const color = () => dot_color(props.status());
  const pinned = () => props.panel_pinned?.() ?? false;

  const dot_shadow = () => {
    if (hovered()) {
      return `0 0 0 3px ${color()}44, 0 0 6px ${color()}88`;
    }
    if (pinned()) {
      return `0 0 0 2px ${color()}88`;
    }
    return "none";
  };

  return (
    <button
      type="button"
      style={{
        display: "flex",
        "align-items": "center",
        "justify-content": "center",
        width: "22px",
        height: "22px",
        "border-radius": "50%",
        "flex-shrink": "0",
        cursor: "pointer",
        background: hovered()
          ? "rgba(255,255,255,0.08)"
          : pinned()
            ? "rgba(255,255,255,0.05)"
            : "transparent",
        transition: "background 0.15s ease",
        border: "none",
        padding: "0",
        font: "inherit",
        color: "inherit",
      }}
      title={
        pinned()
          ? `${props.status()} — panel pinned, click to unpin`
          : `${props.status()} — click to pin panel`
      }
      onMouseEnter={() => {
        set_hovered(true);
        props.on_show_panel?.();
      }}
      onMouseLeave={() => {
        set_hovered(false);
        props.on_hide_panel?.();
      }}
      onClick={(e) => {
        e.stopPropagation();
        props.on_toggle_panel?.();
      }}
    >
      <span
        style={{
          width: "10px",
          height: "10px",
          "border-radius": "50%",
          background: color(),
          "box-shadow": dot_shadow(),
          transform: hovered() ? "scale(1.25)" : pinned() ? "scale(1.1)" : "scale(1)",
          transition: "box-shadow 0.15s ease, transform 0.15s ease",
        }}
      />
    </button>
  );
};
