import { type Accessor, type Component, createSignal, For } from "solid-js";
import { parseISODate } from "../lib/date";

const MINUTE_OPTIONS = [0, 1, 5, 15, 30, 60] as const;
export type TimeUnit = "s" | "m" | "h" | "d";
const UNIT_TO_MINUTES: Record<TimeUnit, number> = { s: 1 / 60, m: 1, h: 60, d: 1440 };

interface LabelTimePickerProps {
  selectedMinutes: Accessor<number>;
  setSelectedMinutes: (m: number) => void;
  fillFromLast: Accessor<boolean>;
  setFillFromLast: (v: boolean) => void;
  lastLabel: Accessor<{ end_ts: string } | null | undefined>;
}

export const LabelTimePicker: Component<LabelTimePickerProps> = (props) => {
  const [customActive, setCustomActive] = createSignal(false);
  const [customValue, setCustomValue] = createSignal("");
  const [customUnit, setCustomUnit] = createSignal<TimeUnit>("m");

  function selectPreset(m: number) {
    props.setSelectedMinutes(m);
    setCustomActive(false);
    setCustomValue("");
    props.setFillFromLast(false);
  }

  function applyCustom(raw: string, unit: TimeUnit) {
    const n = parseFloat(raw);
    if (!Number.isNaN(n) && n >= 0) {
      props.setSelectedMinutes(n * UNIT_TO_MINUTES[unit]);
      setCustomActive(true);
      props.setFillFromLast(false);
    }
  }

  function stepCustom(delta: number) {
    const cur = parseFloat(customValue()) || 0;
    const next = Math.max(0, cur + delta);
    const v = String(next);
    setCustomValue(v);
    applyCustom(v, customUnit());
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
            !customActive() && !props.fillFromLast() && props.selectedMinutes() === m;
          return (
            <button
              type="button"
              onClick={() => selectPreset(m)}
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
        const ll = props.lastLabel();
        if (!ll?.end_ts) {
          return null;
        }
        const ago = Math.round(
          (Date.now() - parseISODate(ll.end_ts).getTime()) / 60_000,
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
              props.setFillFromLast(true);
              setCustomActive(false);
            }}
            style={{
              padding: "2px 7px",
              "border-radius": "10px",
              border: "1px solid var(--border)",
              background: props.fillFromLast() ? "var(--text-muted)" : "var(--surface)",
              color: props.fillFromLast() ? "var(--bg)" : "var(--text-muted)",
              cursor: "pointer",
              "font-size": "0.7rem",
              "font-weight": props.fillFromLast() ? "700" : "500",
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
            value={customValue()}
            onInput={(e) => {
              const v = e.currentTarget.value;
              if (v === "" || /^\d*\.?\d*$/.test(v)) {
                setCustomValue(v);
                applyCustom(v, customUnit());
              } else {
                e.currentTarget.value = customValue();
              }
            }}
            onFocus={() => {
              if (customValue()) {
                applyCustom(customValue(), customUnit());
              }
            }}
            style={{
              width: "28px",
              padding: "2px 4px",
              border: "none",
              background: "transparent",
              color: customActive() ? "var(--text)" : "var(--text-muted)",
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
              onClick={() => stepCustom(1)}
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
              onClick={() => stepCustom(-1)}
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
                  setCustomUnit(u);
                  if (customValue()) {
                    applyCustom(customValue(), u);
                  }
                }}
                style={{
                  padding: "2px 5px",
                  border: "none",
                  "border-right": u !== "d" ? "1px solid var(--border)" : "none",
                  background:
                    customUnit() === u ? "var(--text-muted)" : "var(--surface)",
                  color: customUnit() === u ? "var(--bg)" : "var(--text-muted)",
                  cursor: "pointer",
                  "font-size": "0.65rem",
                  "font-weight": customUnit() === u ? "700" : "500",
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
