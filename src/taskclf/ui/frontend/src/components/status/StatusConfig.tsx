import { type Accessor, type Component, createMemo } from "solid-js";
import { truncPath } from "../../lib/format";
import type { TrayState } from "../../lib/ws";
import { StatusRow } from "../ui/StatusRow";
import { StatusSection } from "../ui/StatusSection";

export const StatusConfig: Component<{
  trayState: Accessor<TrayState>;
}> = (props) => {
  const t = () => props.trayState();

  const summary = createMemo(() => (t().dev_mode ? "dev" : "prod"));

  return (
    <StatusSection title="Config" summary={summary()}>
      <StatusRow
        label="data_dir"
        value={truncPath(t().data_dir)}
        dim
        mono
        tooltip="Root directory for all taskclf data"
      />
      <StatusRow
        label="ui_port"
        value={String(t().ui_port)}
        dim
        tooltip="Port the UI server is listening on"
      />
      <StatusRow
        label="dev_mode"
        value={t().dev_mode ? "yes" : "no"}
        dim
        tooltip="Whether the system is running in development mode"
      />
      <StatusRow
        label="labels_saved"
        value={String(t().labels_saved_count)}
        tooltip="Total number of label spans saved to disk"
      />
    </StatusSection>
  );
};
