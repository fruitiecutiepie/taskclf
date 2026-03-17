import { type Accessor, type Component, createMemo, For, Show } from "solid-js";
import type { StatusEvent } from "../../lib/ws";
import { StatusRow } from "../ui/StatusRow";
import { StatusSection } from "../ui/StatusSection";

export const StatusActivityWatch: Component<{
  status: Accessor<StatusEvent>;
}> = (props) => {
  const s = () => props.status();

  const summary = createMemo(() => (s().aw_connected ? "connected" : "disconnected"));
  const summary_color = createMemo(() => (s().aw_connected ? "#22c55e" : "#ef4444"));

  const app_counts = createMemo(() => {
    const v = s();
    if (!v.last_app_counts) {
      return [];
    }
    return Object.entries(v.last_app_counts).sort(([, a], [, b]) => b - a);
  });

  return (
    <StatusSection
      title="ActivityWatch"
      summary={summary()}
      summary_color={summary_color()}
    >
      <StatusRow
        label="connection"
        value={s().aw_connected ? "connected" : "disconnected"}
        color={s().aw_connected ? "#22c55e" : "#ef4444"}
        tooltip="Whether ActivityWatch is reachable"
      />
      <StatusRow
        label="host"
        value={s().aw_host || "—"}
        dim
        mono
        tooltip="ActivityWatch server hostname"
      />
      <StatusRow
        label="bucket_id"
        value={s().aw_bucket_id ?? "—"}
        dim
        mono
        tooltip="The ActivityWatch bucket being monitored for events"
      />
      <StatusRow
        label="last_events"
        value={String(s().last_event_count)}
        dim
        tooltip="Number of events returned from the last ActivityWatch poll"
      />
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
            app distribution (last poll)
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
