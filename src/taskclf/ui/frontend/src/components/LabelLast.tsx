import { type Accessor, type Component, Show } from "solid-js";
import { time_ago } from "../lib/date";
import { LABEL_COLORS } from "../lib/labelColors";
import { label_entry_is_open_ended } from "../lib/labelTimeline";

type LabelLastProps = {
  last_label: Accessor<
    | {
        label: string;
        start_ts: string;
        end_ts: string;
        extend_forward?: boolean;
      }
    | null
    | undefined
  >;
  is_current?: Accessor<boolean>;
};

const LabelLastContent: Component<{
  ll: Accessor<{
    label: string;
    start_ts: string;
    end_ts: string;
    extend_forward?: boolean;
  }>;
  is_current?: Accessor<boolean>;
}> = (props) => {
  const is_current = () =>
    props.is_current?.() ?? label_entry_is_open_ended(props.ll());

  return (
    <>
      <Show when={is_current()} fallback={"Last:"}>
        Current:
      </Show>{" "}
      <span
        style={{
          color: LABEL_COLORS[props.ll().label] ?? "var(--text)",
          "font-weight": "600",
        }}
      >
        {props.ll().label}
      </span>{" "}
      <Show when={is_current()} fallback={time_ago(props.ll().end_ts)}>
        since {time_ago(props.ll().start_ts)}
      </Show>
    </>
  );
};

export const LabelLast: Component<LabelLastProps> = (props) => (
  <div
    style={{
      "text-align": "center",
      "font-size": "0.65rem",
      color: "var(--text-muted)",
      "margin-top": "6px",
      "margin-bottom": "2px",
      "padding-top": "4px",
      "border-top": "1px solid var(--border)",
    }}
  >
    <Show
      when={props.last_label()}
      fallback={<span style={{ color: "var(--text-muted)" }}>No labels yet</span>}
    >
      {(ll) => <LabelLastContent ll={ll} is_current={props.is_current} />}
    </Show>
  </div>
);
