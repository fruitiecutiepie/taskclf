import { type Accessor, type Component, createMemo } from "solid-js";
import { path_trunc } from "../../lib/format";
import type { TrayState } from "../../lib/ws";
import { StatusRow } from "../ui/StatusRow";
import { StatusSection } from "../ui/StatusSection";

const STATUS_ROW_DEFAULTS = {
  color: undefined,
  dim: undefined,
  mono: undefined,
  tooltip: undefined,
} as const;

export const StatusConfig: Component<{
  tray_state: Accessor<TrayState>;
}> = (props) => {
  const t = () => props.tray_state();

  const summary = createMemo(() => (t().dev_mode ? "dev" : "prod"));

  return (
    <StatusSection
      title="Config"
      summary={summary()}
      summary_color={undefined}
      default_open={undefined}
    >
      <StatusRow
        {...STATUS_ROW_DEFAULTS}
        label="data_dir"
        value={path_trunc(t().data_dir)}
        dim
        mono
        tooltip="Root directory for all taskclf data"
      />
      <StatusRow
        {...STATUS_ROW_DEFAULTS}
        label="ui_port"
        value={String(t().ui_port)}
        dim
        tooltip="Port the UI server is listening on"
      />
      <StatusRow
        {...STATUS_ROW_DEFAULTS}
        label="dev_mode"
        value={t().dev_mode ? "yes" : "no"}
        dim
        tooltip="Whether the system is running in development mode"
      />
      <StatusRow
        {...STATUS_ROW_DEFAULTS}
        label="labels_saved"
        value={String(t().labels_saved_count)}
        tooltip="Total number of label spans saved to disk"
      />
    </StatusSection>
  );
};
