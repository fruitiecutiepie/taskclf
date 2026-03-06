import { type Accessor, type Component, createSignal } from "solid-js";
import type { ConnectionStatus } from "../lib/ws";
import { dotColor } from "./StatePanel";

export const LiveBadgeConnectionStatus: Component<{
  status: Accessor<ConnectionStatus>;
  onTogglePanel?: () => void;
  onShowPanel?: () => void;
  onHidePanel?: () => void;
}> = (props) => {
  const [hovered, setHovered] = createSignal(false);
  const color = () => dotColor(props.status());

  return (
    <span
      style={{
        display: "flex",
        "align-items": "center",
        "justify-content": "center",
        width: "22px",
        height: "22px",
        "border-radius": "50%",
        "flex-shrink": "0",
        cursor: "pointer",
        background: hovered() ? "rgba(255,255,255,0.08)" : "transparent",
        transition: "background 0.15s ease",
      }}
      title={`${props.status()} — click to pin panel`}
      onMouseEnter={() => {
        setHovered(true);
        props.onShowPanel?.();
      }}
      onMouseLeave={() => {
        setHovered(false);
        props.onHidePanel?.();
      }}
      onClick={(e) => {
        e.stopPropagation();
        props.onTogglePanel?.();
      }}
    >
      <span
        style={{
          width: "10px",
          height: "10px",
          "border-radius": "50%",
          background: color(),
          "box-shadow": hovered()
            ? `0 0 0 3px ${color()}44, 0 0 6px ${color()}88`
            : "none",
          transform: hovered() ? "scale(1.25)" : "scale(1)",
          transition: "box-shadow 0.15s ease, transform 0.15s ease",
        }}
      />
    </span>
  );
};
