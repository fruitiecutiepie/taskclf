import { type Component, createMemo, createSignal, For, Show } from "solid-js";
import { iso_date_parse } from "../lib/date";
import { LABEL_COLORS } from "../lib/labelColors";

export type OverwritePending = {
  label: string;
  start: string;
  end: string;
  conflicts: { start_ts: string; end_ts: string; label: string }[];
  confidence: number;
  extend_forward: boolean;
};

type LabelOverwriteProps = {
  pending: OverwritePending;
  on_confirm: () => void;
  on_keep_all: () => void;
  on_cancel: () => void;
};

const fmt = (d: Date) =>
  d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });

export const LabelOverwrite: Component<LabelOverwriteProps> = (props) => {
  const [expanded, set_expanded] = createSignal(false);

  const sorted = createMemo(() =>
    [...props.pending.conflicts].sort((a, b) =>
      a.start_ts < b.start_ts ? -1 : a.start_ts > b.start_ts ? 1 : 0,
    ),
  );

  const unique_labels = createMemo(() => {
    const seen = new Set<string>();
    const out: string[] = [];
    for (const c of sorted()) {
      if (!seen.has(c.label)) {
        seen.add(c.label);
        out.push(c.label);
      }
    }
    return out;
  });

  const affected_range = createMemo(() => {
    const new_start = iso_date_parse(props.pending.start);
    const new_end = iso_date_parse(props.pending.end);
    return `${fmt(new_start)}\u2013${fmt(new_end)}`;
  });

  return (
    <div
      style={{
        "text-align": "center",
        "font-size": "0.8rem",
        "margin-top": "8px",
        "margin-bottom": "8px",
        color: "var(--danger)",
      }}
    >
      <span>
        Overlaps {sorted().length === 1 ? "" : `${sorted().length} labels: `}
        <For each={unique_labels()}>
          {(lbl, i) => {
            const color = LABEL_COLORS[lbl] ?? "var(--text)";
            return (
              <>
                {i() > 0 && ", "}
                <span style={{ color, "font-weight": "700" }}>{lbl}</span>
              </>
            );
          }}
        </For>
      </span>

      <div
        style={{
          color: "var(--text-muted)",
          "font-size": "0.7rem",
          "margin-top": "2px",
        }}
      >
        ({affected_range()} →{" "}
        <span
          style={{
            color: LABEL_COLORS[props.pending.label] ?? "var(--text)",
            "font-weight": "700",
          }}
        >
          {props.pending.label}
        </span>
        )
      </div>

      <Show when={sorted().length > 1}>
        <button
          type="button"
          onClick={() => set_expanded((v) => !v)}
          style={{
            background: "none",
            border: "none",
            color: "var(--text-muted)",
            cursor: "pointer",
            "font-size": "0.65rem",
            "text-decoration": "underline",
            padding: "2px 0",
          }}
        >
          {expanded() ? "hide details" : "show details"}
        </button>
      </Show>

      <Show when={sorted().length === 1 || expanded()}>
        <div
          style={{
            "font-size": "0.7rem",
            color: "var(--text-muted)",
            "margin-top": "2px",
          }}
        >
          <For each={sorted()}>
            {(c) => {
              const color = LABEL_COLORS[c.label] ?? "var(--text)";
              const cs = iso_date_parse(c.start_ts);
              const ce = iso_date_parse(c.end_ts);
              return (
                <div>
                  <span style={{ color, "font-weight": "600" }}>{c.label}</span>{" "}
                  {fmt(cs)}
                  {"\u2013"}
                  {fmt(ce)}
                </div>
              );
            }}
          </For>
        </div>
      </Show>

      <div
        style={{
          display: "flex",
          "justify-content": "center",
          gap: "8px",
          "margin-top": "4px",
        }}
      >
        <button
          type="button"
          onClick={props.on_confirm}
          style={{
            padding: "2px 10px",
            "border-radius": "6px",
            border: "1px solid var(--danger)",
            background: "var(--danger)",
            color: "#fff",
            cursor: "pointer",
            "font-size": "0.7rem",
            "font-weight": "600",
          }}
        >
          Overwrite All
        </button>
        <button
          type="button"
          onClick={props.on_keep_all}
          style={{
            padding: "2px 10px",
            "border-radius": "6px",
            border: "1px solid var(--accent, #6366f1)",
            background: "var(--accent, #6366f1)",
            color: "#fff",
            cursor: "pointer",
            "font-size": "0.7rem",
            "font-weight": "600",
          }}
        >
          Keep All
        </button>
        <button
          type="button"
          onClick={props.on_cancel}
          style={{
            padding: "2px 10px",
            "border-radius": "6px",
            border: "1px solid var(--border)",
            background: "var(--surface)",
            color: "var(--text-muted)",
            cursor: "pointer",
            "font-size": "0.7rem",
            "font-weight": "600",
          }}
        >
          Cancel
        </button>
      </div>
    </div>
  );
};
