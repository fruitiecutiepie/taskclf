import { createSignal, createResource, For, Show, type Component } from "solid-js";
import { createLabel, fetchCoreLabels } from "../lib/api";

const LABEL_COLORS: Record<string, string> = {
  Build: "#6366f1",
  Debug: "#f59e0b",
  Review: "#8b5cf6",
  Write: "#3b82f6",
  ReadResearch: "#14b8a6",
  Communicate: "#f97316",
  Meet: "#ec4899",
  BreakIdle: "#6b7280",
};

const MINUTE_OPTIONS = [0, 1, 5, 10, 15, 30, 60] as const;
type TimeUnit = "s" | "m" | "h" | "d";
const UNIT_TO_MINUTES: Record<TimeUnit, number> = { s: 1 / 60, m: 1, h: 60, d: 1440 };

const EXTEND_FWD_KEY = "taskclf:extendForward";
const _LEGACY_KEY = "taskclf:extendPrevious";
function loadExtendForward(): boolean {
  try {
    const v = localStorage.getItem(EXTEND_FWD_KEY) ?? localStorage.getItem(_LEGACY_KEY);
    return v !== "false";
  } catch { return true; }
}

interface LabelGridProps {
  maxHeight?: number;
  onCollapse: () => void;
}

export const LabelGrid: Component<LabelGridProps> = (props) => {
  const [labels] = createResource(fetchCoreLabels);
  const [flash, setFlash] = createSignal<string | null>(null);
  const [selectedMinutes, setSelectedMinutes] = createSignal(0);
  const [customActive, setCustomActive] = createSignal(false);
  const [customValue, setCustomValue] = createSignal("");
  const [customUnit, setCustomUnit] = createSignal<TimeUnit>("m");
  const [extendFwd, setExtendFwd] = createSignal(loadExtendForward());

  function toggleExtendFwd() {
    const next = !extendFwd();
    setExtendFwd(next);
    try { localStorage.setItem(EXTEND_FWD_KEY, String(next)); } catch {}
  }

  function selectPreset(m: number) {
    setSelectedMinutes(m);
    setCustomActive(false);
    setCustomValue("");
  }

  function applyCustom(raw: string, unit: TimeUnit) {
    const n = parseFloat(raw);
    if (!isNaN(n) && n >= 0) {
      setSelectedMinutes(n * UNIT_TO_MINUTES[unit]);
      setCustomActive(true);
    }
  }

  function stepCustom(delta: number) {
    const cur = parseFloat(customValue()) || 0;
    const next = Math.max(0, cur + delta);
    const v = String(next);
    setCustomValue(v);
    applyCustom(v, customUnit());
  }

  async function labelNow(label: string) {
    const mins = selectedMinutes();
    const now = new Date();
    const start = new Date(now.getTime() - Math.max(mins * 60_000, 1_000));
    try {
      await createLabel({
        start_ts: start.toISOString().slice(0, -1),
        end_ts: now.toISOString().slice(0, -1),
        label,
        extend_forward: extendFwd(),
      });
      setFlash(label);
      setTimeout(() => {
        setFlash(null);
        props.onCollapse();
      }, 1500);
    } catch (err: any) {
      setFlash(`Error: ${err.message}`);
      setTimeout(() => setFlash(null), 3000);
    }
  }

  return (
    <div
      style={{
        padding: "8px 12px 12px",
        "border-top": "1px solid var(--border)",
        ...(props.maxHeight != null
          ? { "max-height": `${props.maxHeight}px`, "overflow-y": "auto" }
          : {}),
      }}
    >
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
          {(m) => (
            <button
              onClick={() => selectPreset(m)}
              style={{
                padding: "2px 7px",
                "border-radius": "10px",
                border: "1px solid var(--border)",
                background:
                  !customActive() && selectedMinutes() === m
                    ? "var(--text-muted)"
                    : "var(--surface)",
                color:
                  !customActive() && selectedMinutes() === m
                    ? "var(--bg)"
                    : "var(--text-muted)",
                cursor: "pointer",
                "font-size": "0.7rem",
                "font-weight":
                  !customActive() && selectedMinutes() === m ? "700" : "500",
                "line-height": "1.4",
                transition: "all 0.1s ease",
              }}
            >
              {m === 0 ? "now" : `${m}m`}
            </button>
          )}
        </For>
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
                if (customValue()) applyCustom(customValue(), customUnit());
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
                  onClick={() => {
                    setCustomUnit(u);
                    if (customValue()) applyCustom(customValue(), u);
                  }}
                  style={{
                    padding: "2px 5px",
                    border: "none",
                    "border-right": u !== "d" ? "1px solid var(--border)" : "none",
                    background: customUnit() === u ? "var(--text-muted)" : "var(--surface)",
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

      <div
        onClick={toggleExtendFwd}
        style={{
          display: "flex",
          "align-items": "center",
          "justify-content": "center",
          gap: "4px",
          "margin-bottom": "6px",
          "font-size": "0.7rem",
          color: "var(--text-muted)",
          cursor: "pointer",
          "user-select": "none",
        }}
      >
        <div
          style={{
            width: "12px",
            height: "12px",
            "border-radius": "3px",
            border: `1.5px solid ${extendFwd() ? "var(--accent)" : "var(--text-muted)"}`,
            background: extendFwd() ? "var(--accent)" : "transparent",
            display: "flex",
            "align-items": "center",
            "justify-content": "center",
            cursor: "pointer",
            "flex-shrink": "0",
            transition: "all 0.15s ease",
          }}
        >
          <Show when={extendFwd()}>
            <svg
              width="8"
              height="8"
              viewBox="0 0 12 12"
              fill="none"
              style={{ display: "block" }}
            >
              <path
                d="M2.5 6L5 8.5L9.5 3.5"
                stroke="var(--bg)"
                stroke-width="2"
                stroke-linecap="round"
                stroke-linejoin="round"
              />
            </svg>
          </Show>
        </div>
        <span
          style={{
            "font-size": "0.7rem",
            color: "var(--text-muted)",
          }}
        >
          Extend until next label
        </span>
      </div>

      <Show when={flash()}>
        <div
          style={{
            "text-align": "center",
            "font-size": "0.8rem",
            "margin-bottom": "8px",
            color: flash()!.startsWith("Error")
              ? "var(--danger)"
              : "var(--success)",
          }}
        >
          {flash()!.startsWith("Error") ? flash() : `Saved: ${flash()}`}
        </div>
      </Show>

      <div
        style={{
          display: "grid",
          "grid-template-columns": "1fr 1fr",
          gap: "6px",
        }}
      >
        <For each={labels() ?? []}>
          {(lbl) => (
            <button
              onClick={() => labelNow(lbl)}
              style={{
                padding: "8px 4px",
                "border-radius": "var(--radius)",
                border: "1px solid var(--border)",
                background: "var(--surface)",
                color: LABEL_COLORS[lbl] ?? "var(--text)",
                cursor: "pointer",
                "font-size": "0.8rem",
                "font-weight": "600",
                "text-align": "center",
                transition: "background 0.1s ease",
              }}
            >
              {lbl}
            </button>
          )}
        </For>
      </div>
    </div>
  );
};
