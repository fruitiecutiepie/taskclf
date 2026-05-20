import { type Accessor, type Component, createMemo, Show } from "solid-js";
import { duration_format, time_format } from "../../lib/format";
import type { StatusEvent } from "../../lib/ws";
import { StatusProgress } from "../ui/StatusProgress";
import { StatusRow } from "../ui/StatusRow";
import { StatusSection } from "../ui/StatusSection";

const STATUS_ROW_DEFAULTS = {
  color: undefined,
  dim: undefined,
  mono: undefined,
  tooltip: undefined,
} as const;

export const StatusActivityMonitor: Component<{
  status: Accessor<StatusEvent>;
}> = (props) => {
  const s = () => props.status();

  const summary = createMemo(() => {
    const v = s();
    return v.current_app ? `${v.state} · ${v.current_app}` : v.state;
  });

  const transition_pct = createMemo(() => {
    const v = s();
    if (!v.candidate_app || !v.transition_threshold_s) {
      return undefined;
    }
    return Math.min(
      100,
      Math.round((v.candidate_duration_s / v.transition_threshold_s) * 100),
    );
  });

  return (
    <StatusSection
      title="Activity Monitor"
      summary={summary()}
      summary_color={undefined}
      default_open
    >
      <StatusRow
        {...STATUS_ROW_DEFAULTS}
        label="state"
        value={s().state}
        tooltip="Current activity state (active, idle, etc.)"
      />
      <StatusRow
        {...STATUS_ROW_DEFAULTS}
        label="current_app"
        value={s().current_app || "—"}
        tooltip="The foreground application currently in use"
      />
      <StatusRow
        {...STATUS_ROW_DEFAULTS}
        label="since"
        value={time_format(s().current_app_since)}
        dim
        tooltip="When the current app became the foreground app"
      />
      <StatusRow
        {...STATUS_ROW_DEFAULTS}
        label="poll_interval"
        value={`${s().poll_seconds}s`}
        dim
      />
      <StatusRow
        {...STATUS_ROW_DEFAULTS}
        label="poll_count"
        value={String(s().poll_count)}
        dim
        tooltip="Total number of polls since startup"
      />
      <StatusRow
        {...STATUS_ROW_DEFAULTS}
        label="last_poll"
        value={time_format(s().last_poll_ts)}
        dim
        tooltip="Timestamp of the most recent poll"
      />
      <StatusRow
        {...STATUS_ROW_DEFAULTS}
        label="uptime"
        value={duration_format(s().uptime_s)}
        dim
        tooltip="How long the activity monitor has been running"
      />
      <Show when={s().candidate_app}>
        {(cand) => (
          <>
            <StatusRow
              {...STATUS_ROW_DEFAULTS}
              label="candidate_app"
              value={cand()}
              color="#eab308"
              tooltip="App that may become the new current app once the transition threshold is reached"
            />
            <StatusRow
              {...STATUS_ROW_DEFAULTS}
              label="candidate_progress"
              value={`${duration_format(s().candidate_duration_s)} / ${duration_format(s().transition_threshold_s)} (${transition_pct()}%)`}
              color="#eab308"
              tooltip="Time spent on the candidate app vs. the threshold required for a transition"
            />
            <StatusProgress pct={transition_pct() ?? 0} color={undefined} />
          </>
        )}
      </Show>
      <Show when={!s().candidate_app}>
        <StatusRow
          {...STATUS_ROW_DEFAULTS}
          label="candidate_app"
          value="none"
          dim
          tooltip="App that may become the new current app once the transition threshold is reached"
        />
      </Show>
    </StatusSection>
  );
};
