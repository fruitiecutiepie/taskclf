import { type Component, createMemo, createSignal, For, Show } from "solid-js";
import { parseISODate } from "../lib/date";
import { LABEL_COLORS } from "../lib/labelColors";

export interface OverwritePending {
  label: string;
  start: string;
  end: string;
  conflicts: { start_ts: string; end_ts: string; label: string }[];
  confidence: number;
  extendForward: boolean;
}

interface LabelOverwriteProps {
  pending: OverwritePending;
  onConfirm: () => void;
  onKeepAll: () => void;
  onCancel: () => void;
}

const fmt = (d: Date) =>
  d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });

export const LabelOverwrite: Component<LabelOverwriteProps> = (props) => {
  const [expanded, setExpanded] = createSignal(false);

  const sorted = createMemo(() =>
    [...props.pending.conflicts].sort((a, b) =>
      a.start_ts < b.start_ts ? -1 : a.start_ts > b.start_ts ? 1 : 0,
    ),
  );

  const uniqueLabels = createMemo(() => {
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

  const affectedRange = createMemo(() => {
    const newStart = parseISODate(props.pending.start);
    const newEnd = parseISODate(props.pending.end);
    return `${fmt(newStart)}\u2013${fmt(newEnd)}`;
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
        <For each={uniqueLabels()}>
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
        ({affectedRange()} →{" "}
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
          onClick={() => setExpanded((v) => !v)}
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
              const cs = parseISODate(c.start_ts);
              const ce = parseISODate(c.end_ts);
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
          onClick={props.onConfirm}
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
          onClick={props.onKeepAll}
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
          onClick={props.onCancel}
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
