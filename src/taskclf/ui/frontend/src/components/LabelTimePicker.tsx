import { type Accessor, type Component, createSignal, For } from "solid-js";
import { iso_date_parse } from "../lib/date";

const MINUTE_OPTIONS = [0, 1, 5, 15, 30, 60] as const;
export type TimeUnit = "s" | "m" | "h" | "d";
const UNIT_TO_MINUTES: Record<TimeUnit, number> = { s: 1 / 60, m: 1, h: 60, d: 1440 };

type LabelTimePickerProps = {
  selected_minutes: Accessor<number>;
  set_selected_minutes: (m: number) => void;
  fill_from_last: Accessor<boolean>;
  set_fill_from_last: (v: boolean) => void;
  last_label: Accessor<{ end_ts: string } | null | undefined>;
};

export const LabelTimePicker: Component<LabelTimePickerProps> = (props) => {
  const [custom_active, set_custom_active] = createSignal(false);
  const [custom_value, set_custom_value] = createSignal("");
  const [custom_unit, set_custom_unit] = createSignal<TimeUnit>("m");

  function preset_select(m: number) {
    props.set_selected_minutes(m);
    set_custom_active(false);
    set_custom_value("");
    props.set_fill_from_last(false);
  }

  function custom_apply(raw: string, unit: TimeUnit) {
    const n = parseFloat(raw);
    if (!Number.isNaN(n) && n >= 0) {
      props.set_selected_minutes(n * UNIT_TO_MINUTES[unit]);
      set_custom_active(true);
      props.set_fill_from_last(false);
    }
  }

  function custom_step(delta: number) {
    const cur = parseFloat(custom_value()) || 0;
    const next = Math.max(0, cur + delta);
    const v = String(next);
    set_custom_value(v);
    custom_apply(v, custom_unit());
  }

  return (
    <div
      style={{
        display: "flex",
        "align-items": "center",
        "justify-content": "center",
        gap: "4px",
        margin: "8px 0",
        "flex-wrap": "wrap",
      }}
    >
      <span
        style={{
          "font-size": "0.7rem",
          color: "var(--text-muted)",
          "margin-right": "2px",
        }}
      >
        Last
      </span>
      <For each={[...MINUTE_OPTIONS]}>
        {(m) => {
          const active = () =>
            !custom_active()
            && !props.fill_from_last()
            && props.selected_minutes() === m;
          return (
            <button
              type="button"
              onClick={() => preset_select(m)}
              style={{
                padding: "2px 7px",
                "border-radius": "10px",
                border: "1px solid var(--border)",
                background: active() ? "var(--text-muted)" : "var(--surface)",
                color: active() ? "var(--bg)" : "var(--text-muted)",
                cursor: "pointer",
                "font-size": "0.7rem",
                "font-weight": active() ? "700" : "500",
                "line-height": "1.4",
                transition: "all 0.1s ease",
              }}
            >
              {m === 0 ? "now" : `${m}m`}
            </button>
          );
        }}
      </For>
      {(() => {
        const ll = props.last_label();
        if (!ll?.end_ts) {
          return null;
        }
        const ago = Math.round(
          (Date.now() - iso_date_parse(ll.end_ts).getTime()) / 60_000,
        );
        if (ago < 1) {
          return null;
        }
        const label =
          ago >= 60
            ? `gap ${Math.floor(ago / 60)}h${ago % 60 ? `${ago % 60}m` : ""}`
            : `gap ${ago}m`;
        return (
          <button
            type="button"
            onClick={() => {
              props.set_fill_from_last(true);
              set_custom_active(false);
            }}
            style={{
              padding: "2px 7px",
              "border-radius": "10px",
              border: "1px solid var(--border)",
              background: props.fill_from_last()
                ? "var(--text-muted)"
                : "var(--surface)",
              color: props.fill_from_last() ? "var(--bg)" : "var(--text-muted)",
              cursor: "pointer",
              "font-size": "0.7rem",
              "font-weight": props.fill_from_last() ? "700" : "500",
              "line-height": "1.4",
              transition: "all 0.1s ease",
            }}
          >
            {label}
          </button>
        );
      })()}
      <div
        style={{
          display: "flex",
          "align-items": "center",
          gap: "2px",
          "margin-left": "2px",
        }}
      >
        <div
          style={{
            display: "flex",
            "align-items": "center",
            "border-radius": "6px",
            border: "1px solid var(--border)",
            background: "var(--surface)",
            overflow: "hidden",
          }}
        >
          <input
            type="text"
            inputmode="decimal"
            placeholder="#"
            value={custom_value()}
            onInput={(e) => {
              const v = e.currentTarget.value;
              if (v === "" || /^\d*\.?\d*$/.test(v)) {
                set_custom_value(v);
                custom_apply(v, custom_unit());
              } else {
                e.currentTarget.value = custom_value();
              }
            }}
            onFocus={() => {
              if (custom_value()) {
                custom_apply(custom_value(), custom_unit());
              }
            }}
            style={{
              width: "28px",
              padding: "2px 4px",
              border: "none",
              background: "transparent",
              color: custom_active() ? "var(--text)" : "var(--text-muted)",
              "font-size": "0.7rem",
              "text-align": "center",
              outline: "none",
            }}
          />
          <div
            style={{
              display: "flex",
              "flex-direction": "column",
              "border-left": "1px solid var(--border)",
            }}
          >
            <button
              type="button"
              onClick={() => custom_step(1)}
              style={{
                display: "flex",
                "align-items": "center",
                "justify-content": "center",
                width: "14px",
                height: "10px",
                border: "none",
                "border-bottom": "0.5px solid var(--border)",
                background: "transparent",
                color: "var(--text-muted)",
                cursor: "pointer",
                "font-size": "0.4rem",
                "line-height": "1",
                padding: "0",
              }}
            >
              ▲
            </button>
            <button
              type="button"
              onClick={() => custom_step(-1)}
              style={{
                display: "flex",
                "align-items": "center",
                "justify-content": "center",
                width: "14px",
                height: "10px",
                border: "none",
                background: "transparent",
                color: "var(--text-muted)",
                cursor: "pointer",
                "font-size": "0.4rem",
                "line-height": "1",
                padding: "0",
              }}
            >
              ▼
            </button>
          </div>
        </div>
        <div
          style={{
            display: "flex",
            "border-radius": "6px",
            border: "1px solid var(--border)",
            overflow: "hidden",
          }}
        >
          <For each={["s", "m", "h", "d"] as TimeUnit[]}>
            {(u) => (
              <button
                type="button"
                onClick={() => {
                  set_custom_unit(u);
                  if (custom_value()) {
                    custom_apply(custom_value(), u);
                  }
                }}
                style={{
                  padding: "2px 5px",
                  border: "none",
                  "border-right": u !== "d" ? "1px solid var(--border)" : "none",
                  background:
                    custom_unit() === u ? "var(--text-muted)" : "var(--surface)",
                  color: custom_unit() === u ? "var(--bg)" : "var(--text-muted)",
                  cursor: "pointer",
                  "font-size": "0.65rem",
                  "font-weight": custom_unit() === u ? "700" : "500",
                  "line-height": "1.4",
                  transition: "all 0.1s ease",
                }}
              >
                {u}
              </button>
            )}
          </For>
        </div>
      </div>
    </div>
  );
};
