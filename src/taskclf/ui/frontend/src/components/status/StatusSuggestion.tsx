import { type Accessor, type Component, createMemo, Show } from "solid-js";
import { time_format } from "../../lib/format";
import { LABEL_COLORS } from "../../lib/labelColors";
import type { LabelSuggestion } from "../../lib/ws";
import { StatusRow } from "../ui/StatusRow";
import { StatusSection } from "../ui/StatusSection";

const STATUS_ROW_DEFAULTS = {
  color: undefined,
  dim: undefined,
  mono: undefined,
  tooltip: undefined,
} as const;

export const StatusSuggestion: Component<{
  suggestion: Accessor<LabelSuggestion | undefined>;
  pending_count: Accessor<number> | undefined;
}> = (props) => {
  const sug = () => props.suggestion();

  const summary = createMemo(() => {
    const s = sug();
    if (!s) {
      return "";
    }
    return `${s.suggested} ${Math.round(s.confidence * 100)}%`;
  });

  const summary_color = createMemo(() => {
    const s = sug();
    return s ? (LABEL_COLORS[s.suggested] ?? "#e0e0e0") : "#a0a0a0";
  });

  return (
    <Show when={sug()}>
      {(s) => (
        <StatusSection
          title="Active Suggestion"
          summary={summary()}
          summary_color={summary_color()}
          default_open={undefined}
        >
          <StatusRow
            {...STATUS_ROW_DEFAULTS}
            label="suggested"
            value={s().suggested}
            color={LABEL_COLORS[s().suggested] ?? "#e0e0e0"}
            tooltip="Label the model suggests changing to"
          />
          <StatusRow
            {...STATUS_ROW_DEFAULTS}
            label="confidence"
            value={`${Math.round(s().confidence * 100)}%`}
            tooltip="How confident the model is in this suggestion"
          />
          <StatusRow
            {...STATUS_ROW_DEFAULTS}
            label="pending"
            value={`${props.pending_count?.() ?? 1}`}
            tooltip="Number of unresolved model suggestions"
          />
          <StatusRow
            {...STATUS_ROW_DEFAULTS}
            label="reason"
            value={s().reason}
            dim
            tooltip="Why the model is suggesting a label change"
          />
          <StatusRow
            {...STATUS_ROW_DEFAULTS}
            label="old_label"
            value={s().old_label}
            dim
            tooltip="The current label before the suggested change"
          />
          <StatusRow
            {...STATUS_ROW_DEFAULTS}
            label="block"
            value={`${time_format(s().block_start)} → ${time_format(s().block_end)}`}
            dim
            tooltip="Time range this suggestion applies to"
          />
        </StatusSection>
      )}
    </Show>
  );
};
