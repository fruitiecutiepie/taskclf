import { type Accessor, type Component, createMemo, Show } from "solid-js";
import { formatTime } from "../../lib/format";
import { LABEL_COLORS } from "../../lib/labelColors";
import type { Prediction } from "../../lib/ws";
import { StatusRow } from "../ui/StatusRow";
import { StatusSection } from "../ui/StatusSection";

export const StatusPrediction: Component<{
  prediction: Accessor<Prediction | null>;
}> = (props) => {
  const pred = () => props.prediction();

  const summary = createMemo(() => {
    const p = pred();
    if (!p) {
      return "none";
    }
    return `${p.mapped_label} ${Math.round(p.confidence * 100)}%`;
  });

  const summaryColor = createMemo(() => {
    const p = pred();
    return p ? (LABEL_COLORS[p.mapped_label] ?? "#e0e0e0") : "#a0a0a0";
  });

  return (
    <StatusSection
      title={pred()?.provenance === "manual" ? "Last Label" : "Last Prediction"}
      summary={summary()}
      summaryColor={summaryColor()}
      defaultOpen
    >
      <Show
        when={pred()}
        fallback={
          <StatusRow
            label="status"
            value="no prediction yet"
            dim
            tooltip="Waiting for the first model prediction"
          />
        }
      >
        {(p) => (
          <>
            <StatusRow
              label="provenance"
              value={p().provenance ?? "unknown"}
              dim
              tooltip="How this label was determined — model prediction or manual assignment"
            />
            <StatusRow
              label="label"
              value={p().label}
              color={LABEL_COLORS[p().label] ?? "#e0e0e0"}
              tooltip="Raw label output from the model"
            />
            <StatusRow
              label="mapped_label"
              value={p().mapped_label}
              color={LABEL_COLORS[p().mapped_label] ?? "#e0e0e0"}
              tooltip="Label after applying any label-mapping rules"
            />
            <StatusRow
              label="confidence"
              value={`${Math.round(p().confidence * 100)}%`}
              color={p().confidence >= 0.5 ? "#22c55e" : "#ef4444"}
              tooltip="Model's confidence in the prediction (higher is better)"
            />
            <StatusRow
              label="ts"
              value={formatTime(p().ts)}
              dim
              tooltip="When this prediction was made"
            />
            <Show when={p().current_app}>
              {(app) => (
                <StatusRow
                  label="trigger_app"
                  value={app()}
                  dim
                  tooltip="The app that was active when this prediction was triggered"
                />
              )}
            </Show>
          </>
        )}
      </Show>
    </StatusSection>
  );
};
