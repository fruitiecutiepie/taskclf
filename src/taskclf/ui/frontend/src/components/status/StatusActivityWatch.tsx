import { type Accessor, type Component, createMemo, For, Show } from "solid-js";
import type { StatusEvent } from "../../lib/ws";
import { ActivitySourceSetupCallout } from "../ActivitySourceSetupCallout";
import { StatusRow } from "../ui/StatusRow";
import { StatusSection } from "../ui/StatusSection";

export const StatusActivityWatch: Component<{
  status: Accessor<StatusEvent>;
}> = (props) => {
  const s = () => props.status();
  const provider = () => s().activity_provider;

  const summary = createMemo(() => provider().state.replace("_", " "));
  const summary_color = createMemo(() => {
    if (provider().state === "ready") {
      return "#22c55e";
    }
    if (provider().state === "setup_required") {
      return "#f59e0b";
    }
    return "#a3a3a3";
  });

  const app_counts = createMemo(() => {
    const breakdown = provider().last_sample_breakdown;
    if (!breakdown) {
      return [];
    }
    return Object.entries(breakdown).sort(([, a], [, b]) => b - a);
  });

  return (
    <StatusSection
      title="Activity Source"
      summary={summary()}
      summary_color={summary_color()}
      default_open={provider().state !== "ready"}
    >
      <StatusRow
        label="provider"
        value={provider().provider_name}
        tooltip="Configured activity source backing live monitoring"
      />
      <StatusRow
        label="state"
        value={provider().state}
        color={summary_color()}
        tooltip="Whether the configured activity source is ready for live summaries"
      />
      <StatusRow
        label="endpoint"
        value={provider().endpoint || "—"}
        dim
        mono
        tooltip="Configured endpoint for the current activity source"
      />
      <StatusRow
        label="source_id"
        value={provider().source_id ?? "—"}
        dim
        mono
        tooltip="Resolved source identifier used for live monitoring"
      />
      <StatusRow
        label="last_sample_count"
        value={String(provider().last_sample_count)}
        dim
        tooltip="Number of source events returned from the last successful sample"
      />
      <Show when={provider().state === "setup_required"}>
        <div style={{ "margin-top": "6px" }}>
          <ActivitySourceSetupCallout provider={provider()} compact />
        </div>
      </Show>
      <Show when={app_counts().length > 0}>
        <div
          style={{
            "margin-top": "2px",
            "padding-left": "2px",
          }}
        >
          <span
            style={{
              color: "#9a9a9a",
              "font-size": "0.58rem",
              "text-transform": "uppercase",
            }}
          >
            app distribution (last sample)
          </span>
          <For each={app_counts()}>
            {([app, count]) => (
              <StatusRow label={`  ${app}`} value={String(count)} dim mono />
            )}
          </For>
        </div>
      </Show>
    </StatusSection>
  );
};
