import { type Accessor, type Component, createMemo, Show } from "solid-js";
import { time_format } from "../../lib/format";
import { dot_color } from "../../lib/labelColors";
import type { ConnectionStatus, WSStats } from "../../lib/ws";
import { StatusRow } from "../ui/StatusRow";
import { StatusSection } from "../ui/StatusSection";

export const StatusWebSocket: Component<{
  status: Accessor<ConnectionStatus>;
  ws_stats: Accessor<WSStats>;
}> = (props) => {
  const stats = () => props.ws_stats();

  const summary = createMemo(() => props.status());
  const summary_color = createMemo(() => dot_color(props.status()));

  return (
    <StatusSection
      title="WebSocket"
      summary={summary()}
      summary_color={summary_color()}
    >
      <StatusRow
        label="status"
        value={props.status()}
        color={dot_color(props.status())}
        tooltip="Current WebSocket connection state"
      />
      <StatusRow
        label="messages"
        value={`${stats().message_count} total`}
        dim
        tooltip="Total messages received since page load"
      />
      <StatusRow
        label="breakdown"
        value={`st:${stats().status_count} pred:${stats().prediction_count} tray:${stats().tray_state_count} sug:${stats().suggestion_count}`}
        dim
        mono
        tooltip="Count of each message type: status, prediction, tray state, suggestion"
      />
      <StatusRow
        label="last_received"
        value={time_format(stats().last_message_at)}
        dim
        tooltip="When the last WebSocket message arrived"
      />
      <StatusRow
        label="reconnects"
        value={String(stats().reconnect_count)}
        dim
        tooltip="Number of times the WebSocket has reconnected"
      />
      <Show when={stats().connected_since}>
        <StatusRow
          label="connected_since"
          value={time_format(stats().connected_since)}
          dim
          tooltip="When the current WebSocket session was established"
        />
      </Show>
    </StatusSection>
  );
};
