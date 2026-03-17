import {
  type Accessor,
  type Component,
  createEffect,
  createResource,
  createSignal,
  For,
  on,
  Show,
} from "solid-js";
import { core_labels_list, label_create, labels_list } from "../lib/api";
import { parseISODate } from "../lib/date";
import { LABEL_COLORS } from "../lib/labelColors";
import { labelOverwritePendingUpdGet } from "../lib/labelOverwritePendingUpdGet";
import type { Prediction } from "../lib/ws";
import { ActivitySummary } from "./ActivitySummary";
import { LabelConfidence } from "./LabelConfidence";
import { LabelExtendToggle } from "./LabelExtendToggle";
import { LabelFlash } from "./LabelFlash";
import { LabelLast } from "./LabelLast";
import { LabelOverwrite, type OverwritePending } from "./LabelOverwrite";
import { LabelTimePicker } from "./LabelTimePicker";

const EXTEND_FWD_KEY = "taskclf:extendForward";
const _LEGACY_KEY = "taskclf:extendPrevious";
function extend_forward_pref_read(): boolean {
  try {
    const v = localStorage.getItem(EXTEND_FWD_KEY) ?? localStorage.getItem(_LEGACY_KEY);
    return v !== "false";
  } catch {
    return true;
  }
}

interface LabelRecorderProps {
  maxHeight?: number;
  onCollapse: () => void;
  prediction?: Accessor<Prediction | null>;
}

export const LabelRecorder: Component<LabelRecorderProps> = (props) => {
  const [labels] = createResource(core_labels_list);
  const [labelVersion, setLabelVersion] = createSignal(0);
  const [lastLabel] = createResource(labelVersion, async () => {
    try {
      const rows = await labels_list(1);
      return rows.length ? rows[0] : null;
    } catch {
      return null;
    }
  });
  const [flash, setFlash] = createSignal<string | null>(null);
  const [overwritePending, setOverwritePending] = createSignal<OverwritePending | null>(
    null,
  );
  const [selectedMinutes, setSelectedMinutes] = createSignal(0);
  const [extendFwd, setExtendFwd] = createSignal(extend_forward_pref_read());
  const [fillFromLast, setFillFromLast] = createSignal(false);
  const [confPercent, setConfPercent] = createSignal(100);

  createEffect(
    on(
      lastLabel,
      () => {
        if (overwritePending()) {
          setOverwritePending(null);
        }
      },
      { defer: true },
    ),
  );

  createEffect(
    on(
      () => [selectedMinutes(), fillFromLast()] as const,
      () => {
        const pending = overwritePending();
        if (!pending) {
          return;
        }

        const updated = labelOverwritePendingUpdGet(
          pending,
          {
            selectedMinutes: selectedMinutes(),
            fillFromLast: fillFromLast(),
            lastLabelEndTs: lastLabel()?.end_ts ?? null,
            extendFwd: extendFwd(),
          },
          new Date(),
        );

        setOverwritePending(updated);
      },
      { defer: true },
    ),
  );

  function extend_fwd_toggle() {
    const next = !extendFwd();
    setExtendFwd(next);
    try {
      localStorage.setItem(EXTEND_FWD_KEY, String(next));
    } catch {}
  }

  async function labelNow(label: string) {
    const mins = selectedMinutes();
    const now = new Date();
    let start: Date;
    let forceExtendFwd = false;

    const lastEndTs = lastLabel()?.end_ts;
    if (fillFromLast() && lastEndTs) {
      start = parseISODate(lastEndTs);
    } else if (mins === 0) {
      start = now;
      forceExtendFwd = true;
    } else {
      start = new Date(now.getTime() - mins * 60_000);
    }
    const effectiveExtend = forceExtendFwd || extendFwd();
    try {
      await label_create({
        start_ts: start.toISOString(),
        end_ts: now.toISOString(),
        label,
        confidence: confPercent() / 100,
        extend_forward: effectiveExtend,
      });
      setFlash(label);
      setLabelVersion((v) => v + 1);
      setTimeout(() => setFlash(null), 1500);
    } catch (err: unknown) {
      const msg: string = err instanceof Error ? err.message : "";
      const jsonMatch = msg.match(/\{[\s\S]*\}/);
      if (jsonMatch) {
        try {
          const parsed = JSON.parse(jsonMatch[0]);
          const overlap = parsed.detail ?? parsed;
          const spans: { start_ts: string; end_ts: string; label: string }[] =
            overlap.conflicting_spans ?? [];
          if (
            spans.length === 0
            && overlap.conflicting_start_ts
            && overlap.conflicting_end_ts
          ) {
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
        } catch {
          /* fall through to generic error */
        }
      }
      setFlash(`Error: ${msg}`);
      setTimeout(() => setFlash(null), 3000);
    }
  }

  async function overwrite_confirm() {
    const pending = overwritePending();
    if (!pending) {
      return;
    }
    setOverwritePending(null);
    try {
      await label_create({
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
    } catch (err: unknown) {
      setFlash(`Error: ${err instanceof Error ? err.message : "overwrite failed"}`);
      setTimeout(() => setFlash(null), 3000);
    }
  }

  async function keep_all_confirm() {
    const pending = overwritePending();
    if (!pending) {
      return;
    }
    setOverwritePending(null);
    try {
      await label_create({
        start_ts: pending.start,
        end_ts: pending.end,
        label: pending.label,
        confidence: pending.confidence,
        extend_forward: pending.extendForward,
        allow_overlap: true,
      });
      setFlash(pending.label);
      setLabelVersion((v) => v + 1);
      setTimeout(() => setFlash(null), 1500);
    } catch (err: unknown) {
      setFlash(`Error: ${err instanceof Error ? err.message : "keep all failed"}`);
      setTimeout(() => setFlash(null), 3000);
    }
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
      <LabelTimePicker
        selectedMinutes={selectedMinutes}
        setSelectedMinutes={setSelectedMinutes}
        fillFromLast={fillFromLast}
        setFillFromLast={setFillFromLast}
        lastLabel={lastLabel}
      />

      <ActivitySummary minutes={selectedMinutes} prediction={props.prediction} />

      <Show when={selectedMinutes() !== 0 || fillFromLast()}>
        <LabelExtendToggle checked={extendFwd} onToggle={extend_fwd_toggle} />
      </Show>

      <LabelConfidence value={confPercent} onChange={setConfPercent} />

      <Show when={overwritePending()}>
        {(pending) => (
          <LabelOverwrite
            pending={pending()}
            onConfirm={overwrite_confirm}
            onKeepAll={keep_all_confirm}
            onCancel={() => setOverwritePending(null)}
          />
        )}
      </Show>

      <Show when={flash()}>
        <button
          type="button"
          onClick={() => setFlash(null)}
          style={{
            cursor: "pointer",
            background: "none",
            border: "none",
            padding: "0",
            width: "100%",
            color: "inherit",
            font: "inherit",
          }}
        >
          <LabelFlash flash={flash() ?? ""} />
        </button>
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
              type="button"
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

      <LabelLast lastLabel={lastLabel} />
    </div>
  );
};
