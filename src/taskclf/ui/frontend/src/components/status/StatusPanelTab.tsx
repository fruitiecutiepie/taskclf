import type { Accessor, Component, Setter } from "solid-js";

export type PanelTab = "system" | "history" | "training";

const TABS: PanelTab[] = ["system", "history", "training"];

export const StatusPanelTab: Component<{
  active: Accessor<PanelTab>;
  onChange: Setter<PanelTab>;
}> = (props) => (
  <div
    style={{
      display: "flex",
      "margin-bottom": "6px",
      "padding-bottom": "4px",
      "border-bottom": "1px solid #2a2a2a",
      gap: "0",
    }}
  >
    {TABS.map((t) => (
      <button
        type="button"
        onClick={() => props.onChange(t)}
        style={{
          flex: "1",
          padding: "3px 0",
          border: "none",
          background: props.active() === t ? "#333" : "transparent",
          color: props.active() === t ? "#e0e0e0" : "#9a9a9a",
          "font-size": "0.7rem",
          "font-weight": props.active() === t ? "700" : "500",
          "font-family": "inherit",
          cursor: "pointer",
          "border-radius": "6px",
          "text-transform": "capitalize",
          "letter-spacing": "0.02em",
          transition: "all 0.15s ease",
        }}
      >
        {t}
      </button>
    ))}
  </div>
);
