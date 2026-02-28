import { type Accessor, type Component, createResource, For, Show } from "solid-js";
import { fetchLabels } from "../lib/api";
import { LABEL_COLORS } from "./StatePanel";

function fmtTime(iso: string): string {
  try {
    const d = new Date(iso.includes("Z") || iso.includes("+") ? iso : iso + "Z");
    return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  } catch {
    return iso;
  }
}

export const LabelHistory: Component<{
  visible: Accessor<boolean>;
}> = (props) => {
  const [labels] = createResource(props.visible, async (show) => {
    if (!show) return [];
    return fetchLabels(10);
  });

  return (
    <Show when={labels()?.length}>
      <div
        style={{
          background: "var(--surface)",
          border: "1px solid #2a2a2a",
          "border-radius": "8px",
          padding: "6px 8px",
          "font-family": "'SF Mono', 'Fira Code', 'Cascadia Code', monospace",
          "font-size": "0.65rem",
          color: "#d0d0d0",
          "box-shadow": "0 8px 32px rgba(0, 0, 0, 0.6)",
        }}
      >
        <div
          style={{
            "font-size": "0.75rem",
            "font-weight": "700",
            color: "#d0d0d0",
            "margin-bottom": "6px",
            "padding-bottom": "4px",
            "border-bottom": "1px solid #2a2a2a",
            "letter-spacing": "0.02em",
          }}
        >
          Label History
        </div>
        <div style={{ "margin-bottom": "5px" }}>
          <div
            style={{
              "font-size": "0.6rem",
              "font-weight": "700",
              "text-transform": "uppercase",
              "letter-spacing": "0.06em",
              color: "#7a7a7a",
              "margin-bottom": "1px",
              "border-bottom": "1px solid #333",
              "padding-bottom": "1px",
            }}
          >
            Recent
          </div>
          <For each={labels()!}>
          {(lbl) => (
            <div
              style={{
                display: "flex",
                "justify-content": "space-between",
                "align-items": "baseline",
                padding: "1px 0",
                gap: "8px",
              }}
            >
              <span
                style={{
                  display: "flex",
                  "align-items": "center",
                  gap: "6px",
                }}
              >
                <span
                  style={{
                    width: "6px",
                    height: "6px",
                    "border-radius": "50%",
                    background: LABEL_COLORS[lbl.label] ?? "#8a8a8a",
                    "flex-shrink": "0",
                  }}
                />
                <span
                  style={{
                    color: LABEL_COLORS[lbl.label] ?? "#d0d0d0",
                    "font-weight": "600",
                    "font-size": "0.65rem",
                  }}
                >
                  {lbl.label}
                </span>
              </span>
              <span
                style={{
                  color: "#8a8a8a",
                  "font-size": "0.65rem",
                  "white-space": "nowrap",
                }}
              >
                {fmtTime(lbl.start_ts)} – {fmtTime(lbl.end_ts)}
              </span>
            </div>
          )}
        </For>
        </div>
      </div>
    </Show>
  );
};
