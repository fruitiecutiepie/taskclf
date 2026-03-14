import { type Accessor, type Component, createMemo, Show } from "solid-js";
import { formatTime } from "../../lib/format";
import { dotColor } from "../../lib/labelColors";
import type { ConnectionStatus, WSStats } from "../../lib/ws";
import { StatusRow } from "../ui/StatusRow";
import { StatusSection } from "../ui/StatusSection";

export const StatusWebSocket: Component<{
  status: Accessor<ConnectionStatus>;
  wsStats: Accessor<WSStats>;
}> = (props) => {
  const stats = () => props.wsStats();

  const summary = createMemo(() => props.status());
  const summaryColor = createMemo(() => dotColor(props.status()));

  return (
    <StatusSection title="WebSocket" summary={summary()} summaryColor={summaryColor()}>
      <StatusRow
        label="status"
        value={props.status()}
        color={dotColor(props.status())}
        tooltip="Current WebSocket connection state"
      />
      <StatusRow
        label="messages"
        value={`${stats().messageCount} total`}
        dim
        tooltip="Total messages received since page load"
      />
      <StatusRow
        label="breakdown"
        value={`st:${stats().statusCount} pred:${stats().predictionCount} tray:${stats().trayStateCount} sug:${stats().suggestionCount}`}
        dim
        mono
        tooltip="Count of each message type: status, prediction, tray state, suggestion"
      />
      <StatusRow
        label="last_received"
        value={formatTime(stats().lastMessageAt)}
        dim
        tooltip="When the last WebSocket message arrived"
      />
      <StatusRow
        label="reconnects"
        value={String(stats().reconnectCount)}
        dim
        tooltip="Number of times the WebSocket has reconnected"
      />
      <Show when={stats().connectedSince}>
        <StatusRow
          label="connected_since"
          value={formatTime(stats().connectedSince)}
          dim
          tooltip="When the current WebSocket session was established"
        />
      </Show>
    </StatusSection>
  );
};
