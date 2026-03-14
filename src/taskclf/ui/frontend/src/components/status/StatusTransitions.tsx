import { type Accessor, type Component, createMemo, Show } from "solid-js";
import { formatTime } from "../../lib/format";
import type { TrayState } from "../../lib/ws";
import { StatusRow } from "../ui/StatusRow";
import { StatusSection } from "../ui/StatusSection";

export const StatusTransitions: Component<{
  trayState: Accessor<TrayState>;
}> = (props) => {
  const t = () => props.trayState();

  const summary = createMemo(() => String(t().transition_count));

  return (
    <StatusSection title="Transitions" summary={summary()}>
      <StatusRow
        label="total"
        value={String(t().transition_count)}
        tooltip="Total number of app transitions detected this session"
      />
      <Show
        when={t().last_transition}
        fallback={
          <StatusRow
            label="last"
            value="none yet"
            dim
            tooltip="Most recent app transition"
          />
        }
      >
        {(tr) => (
          <>
            <StatusRow
              label="prev → new"
              value={`${tr().prev_app} → ${tr().new_app}`}
              tooltip="The previous and new app in the last transition"
            />
            <StatusRow
              label="block"
              value={`${formatTime(tr().block_start)} → ${formatTime(tr().block_end)}`}
              dim
              tooltip="Time range of the activity block that ended with this transition"
            />
            <StatusRow
              label="fired_at"
              value={formatTime(tr().fired_at)}
              dim
              tooltip="When this transition was triggered"
            />
          </>
        )}
      </Show>
    </StatusSection>
  );
};
