import { type Component, For, Show } from "solid-js";
import type { ActivityProviderStatus } from "../lib/api";

export const ActivitySourceSetupCallout: Component<{
  provider: ActivityProviderStatus;
  compact?: boolean;
}> = (props) => {
  const compact = () => props.compact ?? false;

  return (
    <div
      style={{
        padding: compact() ? "6px 8px" : "8px 10px",
        "border-radius": "8px",
        border: "1px solid color-mix(in srgb, #f59e0b 45%, var(--border))",
        background: "color-mix(in srgb, #f59e0b 10%, transparent)",
        color: "var(--text)",
        display: "flex",
        "flex-direction": "column",
        gap: compact() ? "4px" : "6px",
      }}
    >
      <div
        style={{
          display: "flex",
          "align-items": "baseline",
          "justify-content": "space-between",
          gap: "8px",
          "flex-wrap": "wrap",
        }}
      >
        <strong style={{ "font-size": compact() ? "0.62rem" : "0.68rem" }}>
          {props.provider.setup_title}
        </strong>
        <span
          style={{
            color: "var(--text-muted)",
            "font-size": compact() ? "0.58rem" : "0.62rem",
          }}
        >
          {props.provider.provider_name}
        </span>
      </div>

      <div
        style={{
          color: "var(--text-muted)",
          "font-size": compact() ? "0.58rem" : "0.62rem",
          "line-height": "1.45",
        }}
      >
        {props.provider.setup_message}
      </div>

      <div
        style={{
          display: "flex",
          "flex-direction": "column",
          gap: "3px",
        }}
      >
        <For each={props.provider.setup_steps}>
          {(step, index) => (
            <div
              style={{
                display: "flex",
                gap: "6px",
                "font-size": compact() ? "0.58rem" : "0.62rem",
                color: "var(--text-muted)",
              }}
            >
              <span style={{ color: "var(--text)" }}>{index() + 1}.</span>
              <span>{step}</span>
            </div>
          )}
        </For>
      </div>

      <Show when={props.provider.help_url}>
        <a
          href={props.provider.help_url}
          target="_blank"
          rel="noreferrer"
          style={{
            color: "#fbbf24",
            "font-size": compact() ? "0.58rem" : "0.62rem",
            "text-decoration": "none",
          }}
        >
          Setup help
        </a>
      </Show>
    </div>
  );
};
