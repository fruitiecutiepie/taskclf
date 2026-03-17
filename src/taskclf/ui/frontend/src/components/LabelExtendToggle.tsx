import { type Accessor, type Component, Show } from "solid-js";

type LabelExtendToggleProps = {
  checked: Accessor<boolean>;
  on_toggle: () => void;
};

export const LabelExtendToggle: Component<LabelExtendToggleProps> = (props) => (
  <button
    type="button"
    onClick={props.on_toggle}
    style={{
      display: "flex",
      "align-items": "center",
      "justify-content": "center",
      gap: "4px",
      "margin-bottom": "6px",
      "font-size": "0.7rem",
      color: "var(--text-muted)",
      cursor: "pointer",
      "user-select": "none",
      background: "none",
      border: "none",
      padding: "0",
    }}
  >
    <div
      style={{
        width: "12px",
        height: "12px",
        "border-radius": "3px",
        border: `1.5px solid ${props.checked() ? "var(--accent)" : "var(--text-muted)"}`,
        background: props.checked() ? "var(--accent)" : "transparent",
        display: "flex",
        "align-items": "center",
        "justify-content": "center",
        cursor: "pointer",
        "flex-shrink": "0",
        transition: "all 0.15s ease",
      }}
    >
      <Show when={props.checked()}>
        <svg
          role="img"
          width="8"
          height="8"
          viewBox="0 0 12 12"
          fill="none"
          style={{ display: "block" }}
        >
          <title>Checkmark</title>
          <path
            d="M2.5 6L5 8.5L9.5 3.5"
            stroke="var(--bg)"
            stroke-width="2"
            stroke-linecap="round"
            stroke-linejoin="round"
          />
        </svg>
      </Show>
    </div>
    <span
      style={{
        "font-size": "0.7rem",
        color: "var(--text-muted)",
      }}
    >
      Extend until next label
    </span>
  </button>
);
