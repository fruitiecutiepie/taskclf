import { type Accessor, type Component, createMemo, Show } from "solid-js";
import { path_trunc } from "../../lib/format";
import { LABEL_COLORS } from "../../lib/labelColors";
import type { TrayState } from "../../lib/ws";
import { StatusRow } from "../ui/StatusRow";
import { StatusSection } from "../ui/StatusSection";

export const StatusModel: Component<{
  trayState: Accessor<TrayState>;
}> = (props) => {
  const t = () => props.trayState();

  const summary = createMemo(() => (t().model_loaded ? "loaded" : "not loaded"));
  const summaryColor = createMemo(() => (t().model_loaded ? "#22c55e" : "#ef4444"));

  return (
    <StatusSection title="Model" summary={summary()} summaryColor={summaryColor()}>
      <StatusRow
        label="loaded"
        value={t().model_loaded ? "yes" : "no"}
        color={t().model_loaded ? "#22c55e" : "#ef4444"}
        tooltip="Whether a trained model is currently loaded for inference"
      />
      <Show when={t().model_dir}>
        {(dir) => (
          <StatusRow
            label="model_dir"
            value={path_trunc(dir())}
            dim
            mono
            tooltip="Directory path of the loaded model bundle"
          />
        )}
      </Show>
      <Show when={t().model_schema_hash}>
        {(hash) => (
          <StatusRow
            label="schema_hash"
            value={hash()}
            dim
            mono
            tooltip="Feature schema hash the model was trained with — must match current schema to run inference"
          />
        )}
      </Show>
      <Show when={t().suggested_label}>
        {(label) => (
          <>
            <StatusRow
              label="suggested"
              value={label()}
              color={LABEL_COLORS[label()] ?? "#e0e0e0"}
              tooltip="Label the model suggests for the current activity block"
            />
            <StatusRow
              label="suggestion_conf"
              value={`${Math.round((t().suggested_confidence ?? 0) * 100)}%`}
              tooltip="Confidence of the current label suggestion"
            />
          </>
        )}
      </Show>
      <Show when={!t().suggested_label}>
        <StatusRow
          label="suggested"
          value="none"
          dim
          tooltip="Label the model suggests for the current activity block"
        />
      </Show>
    </StatusSection>
  );
};
