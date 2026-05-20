import { type Accessor, type Component, createMemo, Show } from "solid-js";
import { time_format } from "../../lib/format";
import type { TrayState } from "../../lib/ws";
import { StatusRow } from "../ui/StatusRow";
import { StatusSection } from "../ui/StatusSection";

const STATUS_ROW_DEFAULTS = {
  color: undefined,
  dim: undefined,
  mono: undefined,
  tooltip: undefined,
} as const;

export const StatusTransitions: Component<{
  tray_state: Accessor<TrayState>;
}> = (props) => {
  const t = () => props.tray_state();

  const summary = createMemo(() => String(t().transition_count));

  return (
    <StatusSection
      title="Transitions"
      summary={summary()}
      summary_color={undefined}
      default_open={undefined}
    >
      <StatusRow
        {...STATUS_ROW_DEFAULTS}
        label="total"
        value={String(t().transition_count)}
        tooltip="Total number of app transitions detected this session"
      />
      <Show
        when={t().last_transition}
        fallback={
          <StatusRow
            {...STATUS_ROW_DEFAULTS}
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
              {...STATUS_ROW_DEFAULTS}
              label="prev → new"
              value={`${tr().prev_app} → ${tr().new_app}`}
              tooltip="The previous and new app in the last transition"
            />
            <StatusRow
              {...STATUS_ROW_DEFAULTS}
              label="block"
              value={`${time_format(tr().block_start)} → ${time_format(tr().block_end)}`}
              dim
              tooltip="Time range of the activity block that ended with this transition"
            />
            <StatusRow
              {...STATUS_ROW_DEFAULTS}
              label="fired_at"
              value={time_format(tr().fired_at)}
              dim
              tooltip="When this transition was triggered"
            />
          </>
        )}
      </Show>
    </StatusSection>
  );
};
