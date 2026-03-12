import { createSignal, createResource, For, Show, type Accessor, type Component } from "solid-js";
import { createLabel, fetchCoreLabels, fetchLabels } from "../lib/api";
import type { Prediction } from "../lib/ws";
import { parseISODate, timeAgo } from "../lib/date";
import { LABEL_COLORS } from "../lib/labelColors";
import { ActivitySummary } from "./ActivitySummary";

const MINUTE_OPTIONS = [0, 1, 5, 15, 30, 60] as const;
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
  prediction?: Accessor<Prediction | null>;
}

export const LabelGrid: Component<LabelGridProps> = (props) => {
  const [labels] = createResource(fetchCoreLabels);
  const [labelVersion, setLabelVersion] = createSignal(0);
  const [lastLabel] = createResource(
    labelVersion,
    async () => {
      try {
        const rows = await fetchLabels(1);
        return rows.length ? rows[0] : null;
      } catch { return null; }
    },
  );
  const [flash, setFlash] = createSignal<string | null>(null);
  const [overwritePending, setOverwritePending] = createSignal<{
    label: string;
    start: string;
    end: string;
    conflicts: { start_ts: string; end_ts: string; label: string }[];
    confidence: number;
    extendForward: boolean;
  } | null>(null);
  const [selectedMinutes, setSelectedMinutes] = createSignal(0);
  const [customActive, setCustomActive] = createSignal(false);
  const [customValue, setCustomValue] = createSignal("");
  const [customUnit, setCustomUnit] = createSignal<TimeUnit>("m");
  const [extendFwd, setExtendFwd] = createSignal(loadExtendForward());
  const [fillFromLast, setFillFromLast] = createSignal(false);
  const [confPercent, setConfPercent] = createSignal(100);

  function toggleExtendFwd() {
    const next = !extendFwd();
    setExtendFwd(next);
    try { localStorage.setItem(EXTEND_FWD_KEY, String(next)); } catch {}
  }

  function selectPreset(m: number) {
    setSelectedMinutes(m);
    setCustomActive(false);
    setCustomValue("");
    setFillFromLast(false);
  }

  function applyCustom(raw: string, unit: TimeUnit) {
    const n = parseFloat(raw);
    if (!isNaN(n) && n >= 0) {
      setSelectedMinutes(n * UNIT_TO_MINUTES[unit]);
      setCustomActive(true);
      setFillFromLast(false);
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
    let start: Date;
    let forceExtendFwd = false;

    if (fillFromLast() && lastLabel()?.end_ts) {
      start = parseISODate(lastLabel()!.end_ts);
    } else if (mins === 0) {
      start = now;
      forceExtendFwd = true;
    } else {
      start = new Date(now.getTime() - mins * 60_000);
    }
    const effectiveExtend = forceExtendFwd || extendFwd();
    try {
      await createLabel({
        start_ts: start.toISOString(),
        end_ts: now.toISOString(),
        label,
        confidence: confPercent() / 100,
        extend_forward: effectiveExtend,
      });
      setFlash(label);
      setLabelVersion((v) => v + 1);
      setTimeout(() => setFlash(null), 1500);
    } catch (err: any) {
      const msg: string = err.message ?? "";
      const jsonMatch = msg.match(/\{[\s\S]*\}/);
      if (jsonMatch) {
        try {
          const parsed = JSON.parse(jsonMatch[0]);
          const overlap = parsed.detail ?? parsed;
          const spans: { start_ts: string; end_ts: string; label: string }[] =
            overlap.conflicting_spans ?? [];
          if (spans.length === 0 && overlap.conflicting_start_ts && overlap.conflicting_end_ts) {
            spans.push({
              start_ts: overlap.conflicting_start_ts,
              end_ts: overlap.conflicting_end_ts,
              label: overlap.conflicting_label ?? "unknown",
            });
          }
          if (spans.length > 0) {
            setOverwritePending({
              label,
              start: start.toISOString(),
              end: now.toISOString(),
              conflicts: spans,
              confidence: confPercent() / 100,
              extendForward: effectiveExtend,
            });
            return;
          }
        } catch { /* fall through to generic error */ }
      }
      setFlash(`Error: ${msg}`);
      setTimeout(() => setFlash(null), 3000);
    }
  }

  async function confirmOverwrite() {
    const pending = overwritePending();
    if (!pending) return;
    setOverwritePending(null);
    try {
      await createLabel({
        start_ts: pending.start,
        end_ts: pending.end,
        label: pending.label,
        confidence: pending.confidence,
        extend_forward: pending.extendForward,
        overwrite: true,
      });
      setFlash(pending.label);
      setLabelVersion((v) => v + 1);
      setTimeout(() => setFlash(null), 1500);
    } catch (err: any) {
      setFlash(`Error: ${err.message ?? "overwrite failed"}`);
      setTimeout(() => setFlash(null), 3000);
    }
  }

  function cancelOverwrite() {
    setOverwritePending(null);
  }

  return (
    <div
      style={{
        padding: "8px",
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
          {(m) => {
            const active = () => !customActive() && !fillFromLast() && selectedMinutes() === m;
            return (
              <button
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
          const ll = lastLabel();
          if (!ll?.end_ts) return null;
          const ago = Math.round((Date.now() - parseISODate(ll.end_ts).getTime()) / 60_000);
          if (ago < 1) return null;
          const label = ago >= 60
            ? `gap ${Math.floor(ago / 60)}h${ago % 60 ? `${ago % 60}m` : ""}`
            : `gap ${ago}m`;
          return (
            <button
              onClick={() => {
                setFillFromLast(true);
                setCustomActive(false);
              }}
              style={{
                padding: "2px 7px",
                "border-radius": "10px",
                border: "1px solid var(--border)",
                background: fillFromLast() ? "var(--text-muted)" : "var(--surface)",
                color: fillFromLast() ? "var(--bg)" : "var(--text-muted)",
                cursor: "pointer",
                "font-size": "0.7rem",
                "font-weight": fillFromLast() ? "700" : "500",
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

      <ActivitySummary minutes={selectedMinutes} prediction={props.prediction} />

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

      <div
        style={{
          display: "flex",
          "align-items": "center",
          "justify-content": "center",
          gap: "6px",
          "margin-bottom": "6px",
        }}
      >
        <span
          style={{
            "font-size": "0.7rem",
            color: "var(--text-muted)",
            "flex-shrink": "0",
          }}
        >
          Confidence
        </span>
        <input
          type="range"
          min="0"
          max="100"
          step="5"
          value={confPercent()}
          onInput={(e) => setConfPercent(parseInt(e.currentTarget.value))}
          style={{
            flex: "1",
            height: "4px",
            "max-width": "120px",
            "accent-color": "var(--accent)",
            cursor: "pointer",
          }}
        />
        <span
          style={{
            "font-size": "0.7rem",
            "font-weight": confPercent() < 100 ? "700" : "500",
            color: confPercent() < 100 ? "var(--text)" : "var(--text-muted)",
            "min-width": "30px",
            "text-align": "right",
            "flex-shrink": "0",
          }}
        >
          {confPercent()}%
        </span>
      </div>

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

      <Show when={overwritePending()}>
        {(pending) => {
          const fmt = (d: Date) =>
            d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
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
                Overlaps{" "}
                <For each={pending().conflicts}>
                  {(c, i) => {
                    const color = LABEL_COLORS[c.label] ?? "var(--text)";
                    const cs = parseISODate(c.start_ts);
                    const ce = parseISODate(c.end_ts);
                    return (
                      <>
                        {i() > 0 && ", "}
                        <span style={{ color, "font-weight": "700" }}>{c.label}</span>
                        {" "}{fmt(cs)}{"\u2013"}{fmt(ce)}
                      </>
                    );
                  }}
                </For>
                . Overwrite?
              </span>
              <div
                style={{
                  display: "flex",
                  "justify-content": "center",
                  gap: "8px",
                  "margin-top": "4px",
                }}
              >
                <button
                  onClick={confirmOverwrite}
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
                  Yes
                </button>
                <button
                  onClick={cancelOverwrite}
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
                  No
                </button>
              </div>
            </div>
          );
        }}
      </Show>

      <Show when={flash()}>
        <div
          onClick={() => setFlash(null)}
          style={{
            "text-align": "center",
            "font-size": "0.8rem",
            "margin-top": "8px",
            "margin-bottom": "8px",
            cursor: "pointer",
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
          when={lastLabel()}
          fallback={<span style={{ color: "var(--text-muted)" }}>No labels yet</span>}
        >
          Last:{" "}
          <span
            style={{
              color: LABEL_COLORS[lastLabel()!.label] ?? "var(--text)",
              "font-weight": "600",
            }}
          >
            {lastLabel()!.label}
          </span>{" "}
          {timeAgo(lastLabel()!.end_ts)}
        </Show>
      </div>
    </div>
  );
};
