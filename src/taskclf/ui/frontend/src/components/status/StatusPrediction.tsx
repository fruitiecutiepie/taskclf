import { type Accessor, type Component, createMemo, Show } from "solid-js";
import { time_format } from "../../lib/format";
import { LABEL_COLORS } from "../../lib/labelColors";
import type { Prediction } from "../../lib/ws";
import { StatusRow } from "../ui/StatusRow";
import { StatusSection } from "../ui/StatusSection";

const STATUS_ROW_DEFAULTS = {
  color: undefined,
  dim: undefined,
  mono: undefined,
  tooltip: undefined,
} as const;

export const StatusPrediction: Component<{
  prediction: Accessor<Prediction | undefined>;
}> = (props) => {
  const pred = () => props.prediction();

  const summary = createMemo(() => {
    const p = pred();
    if (!p) {
      return "none";
    }
    return `${p.mapped_label} ${Math.round(p.confidence * 100)}%`;
  });

  const summary_color = createMemo(() => {
    const p = pred();
    return p ? (LABEL_COLORS[p.mapped_label] ?? "#e0e0e0") : "#a0a0a0";
  });

  return (
    <StatusSection
      title={pred()?.provenance === "manual" ? "Last Label" : "Last Prediction"}
      summary={summary()}
      summary_color={summary_color()}
      default_open
    >
      <Show
        when={pred()}
        fallback={
          <StatusRow
            {...STATUS_ROW_DEFAULTS}
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
              {...STATUS_ROW_DEFAULTS}
              label="provenance"
              value={p().provenance ?? "unknown"}
              dim
              tooltip="How this label was determined — model prediction or manual assignment"
            />
            <StatusRow
              {...STATUS_ROW_DEFAULTS}
              label="label"
              value={p().label}
              color={LABEL_COLORS[p().label] ?? "#e0e0e0"}
              tooltip="Raw label output from the model"
            />
            <StatusRow
              {...STATUS_ROW_DEFAULTS}
              label="mapped_label"
              value={p().mapped_label}
              color={LABEL_COLORS[p().mapped_label] ?? "#e0e0e0"}
              tooltip="Label after applying any label-mapping rules"
            />
            <StatusRow
              {...STATUS_ROW_DEFAULTS}
              label="confidence"
              value={`${Math.round(p().confidence * 100)}%`}
              color={p().confidence >= 0.5 ? "#22c55e" : "#ef4444"}
              tooltip="Model's confidence in the prediction (higher is better)"
            />
            <StatusRow
              {...STATUS_ROW_DEFAULTS}
              label="ts"
              value={time_format(p().ts)}
              dim
              tooltip="When this prediction was made"
            />
            <Show when={p().current_app}>
              {(app) => (
                <StatusRow
                  {...STATUS_ROW_DEFAULTS}
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
