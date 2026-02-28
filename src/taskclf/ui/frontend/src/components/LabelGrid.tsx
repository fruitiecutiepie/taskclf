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

const EXTEND_PREV_KEY = "taskclf:extendPrevious";
function loadExtendPrevious(): boolean {
  try { return localStorage.getItem(EXTEND_PREV_KEY) !== "false"; }
  catch { return true; }
}

interface LabelGridProps {
  maxHeight: number;
  onCollapse: () => void;
}

export const LabelGrid: Component<LabelGridProps> = (props) => {
  const [labels] = createResource(fetchCoreLabels);
  const [flash, setFlash] = createSignal<string | null>(null);
  const [selectedMinutes, setSelectedMinutes] = createSignal(1);
  const [customActive, setCustomActive] = createSignal(false);
  const [customValue, setCustomValue] = createSignal("");
  const [customUnit, setCustomUnit] = createSignal<TimeUnit>("m");
  const [extendPrev, setExtendPrev] = createSignal(loadExtendPrevious());

  function toggleExtendPrev() {
    const next = !extendPrev();
    setExtendPrev(next);
    try { localStorage.setItem(EXTEND_PREV_KEY, String(next)); } catch {}
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

  async function labelNow(label: string) {
    const mins = selectedMinutes();
    const now = new Date();
    const start = new Date(now.getTime() - mins * 60_000);
    try {
      await createLabel({
        start_ts: start.toISOString().slice(0, -1),
        end_ts: now.toISOString().slice(0, -1),
        label,
        extend_previous: extendPrev(),
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
        "max-height": `${props.maxHeight}px`,
        "overflow-y": "auto",
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
          <input
            type="number"
            min="0"
            step="any"
            placeholder="#"
            value={customValue()}
            onInput={(e) => {
              const v = e.currentTarget.value;
              setCustomValue(v);
              applyCustom(v, customUnit());
            }}
            onFocus={() => {
              if (customValue()) applyCustom(customValue(), customUnit());
            }}
            style={{
              width: "36px",
              padding: "2px 4px",
              "border-radius": "6px",
              border: `1px solid ${customActive() ? "var(--text-muted)" : "var(--border)"}`,
              background: "var(--surface)",
              color: "var(--text-muted)",
              "font-size": "0.7rem",
              "text-align": "center",
              outline: "none",
            }}
          />
          <select
            value={customUnit()}
            onChange={(e) => {
              const u = e.currentTarget.value as TimeUnit;
              setCustomUnit(u);
              if (customValue()) applyCustom(customValue(), u);
            }}
            style={{
              padding: "2px 2px",
              "border-radius": "6px",
              border: `1px solid ${customActive() ? "var(--text-muted)" : "var(--border)"}`,
              background: "var(--surface)",
              color: "var(--text-muted)",
              "font-size": "0.7rem",
              cursor: "pointer",
              outline: "none",
            }}
          >
            <option value="s">s</option>
            <option value="m">m</option>
            <option value="h">h</option>
            <option value="d">d</option>
          </select>
        </div>
      </div>

      <label
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
        <input
          type="checkbox"
          checked={extendPrev()}
          onChange={toggleExtendPrev}
          style={{ margin: "0", cursor: "pointer" }}
        />
        Fill gap since last label
      </label>

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
