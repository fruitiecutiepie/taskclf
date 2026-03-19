import { type Component, createSignal, type JSX, Show } from "solid-js";

export const StatusSection: Component<{
  title: string;
  summary?: string;
  summary_color?: string;
  default_open?: boolean;
  children: JSX.Element;
}> = (props) => {
  const [open, set_open] = createSignal(props.default_open ?? false);

  return (
    <div style={{ "margin-bottom": "3px" }}>
      <button
        type="button"
        onClick={() => set_open((v) => !v)}
        style={{
          display: "flex",
          "align-items": "center",
          "justify-content": "space-between",
          cursor: "pointer",
          "user-select": "none",
          "font-size": "0.65rem",
          "font-weight": "700",
          "text-transform": "uppercase",
          "letter-spacing": "0.06em",
          color: "#b0b0b0",
          padding: "3px 5px 2px",
          "border-radius": "4px",
          background: "rgba(255, 255, 255, 0.04)",
          border: "none",
          width: "100%",
        }}
      >
        <div style={{ display: "flex", "align-items": "center", gap: "4px" }}>
          <span
            style={{
              display: "inline-block",
              transition: "transform 0.15s ease",
              transform: open() ? "rotate(90deg)" : "rotate(0deg)",
              "font-size": "0.55rem",
              color: "#808080",
            }}
          >
            ▶
          </span>
          <span>{props.title}</span>
        </div>
        <Show when={!open() && props.summary}>
          <span
            style={{
              "font-size": "0.65rem",
              "font-weight": "600",
              color: props.summary_color ?? "#b0b0b0",
              "text-transform": "none",
              "letter-spacing": "normal",
              overflow: "hidden",
              "text-overflow": "ellipsis",
              "white-space": "nowrap",
              "max-width": "140px",
            }}
          >
            {props.summary}
          </span>
        </Show>
      </button>
      <Show when={open()}>
        <div style={{ padding: "2px 0 0 6px" }}>{props.children}</div>
      </Show>
    </div>
  );
};
