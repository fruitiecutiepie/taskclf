import type { Accessor, Component, Setter } from "solid-js";

export type PanelTab = "system" | "history" | "training";

const TABS: PanelTab[] = ["system", "history", "training"];

export const StatusPanelTab: Component<{
  active: Accessor<PanelTab>;
  on_change: Setter<PanelTab>;
  history_pending?: Accessor<boolean>;
  on_history_pending_click?: () => void;
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
        onClick={() => props.on_change(t)}
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
        <span
          style={{
            display: "inline-flex",
            "align-items": "center",
            gap: "4px",
          }}
        >
          {t}
          {t === "history" && props.history_pending?.() && (
            <button
              type="button"
              title="Pending inferred label confirmation"
              onClick={(e) => {
                e.stopPropagation();
                props.on_history_pending_click?.();
              }}
              style={{
                width: "6px",
                height: "6px",
                "border-radius": "999px",
                background: "#f59e0b",
                "box-shadow": "0 0 0 1px rgba(245,158,11,0.35)",
                "flex-shrink": "0",
                border: "none",
                padding: "0",
                cursor: "pointer",
              }}
            />
          )}
        </span>
      </button>
    ))}
  </div>
);
