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
import { iso_date_parse } from "../lib/date";
import { label_overwrite_pending_upd_get } from "../lib/label_overwrite_pending_upd_get";
import { LABEL_COLORS } from "../lib/labelColors";
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

type LabelRecorderProps = {
  max_height?: number;
  on_collapse: () => void;
  prediction?: Accessor<Prediction | null>;
};

export const LabelRecorder: Component<LabelRecorderProps> = (props) => {
  const [labels] = createResource(core_labels_list);
  const [label_version, set_label_version] = createSignal(0);
  const [last_label] = createResource(label_version, async () => {
    try {
      const rows = await labels_list(1);
      return rows.length ? rows[0] : null;
    } catch {
      return null;
    }
  });
  const [flash, set_flash] = createSignal<string | null>(null);
  const [overwrite_pending, set_overwrite_pending] =
    createSignal<OverwritePending | null>(null);
  const [selected_minutes, set_selected_minutes] = createSignal(0);
  const [extend_fwd, set_extend_fwd] = createSignal(extend_forward_pref_read());
  const [fill_from_last, set_fill_from_last] = createSignal(false);
  const [conf_percent, set_conf_percent] = createSignal(100);

  createEffect(
    on(
      last_label,
      () => {
        if (overwrite_pending()) {
          set_overwrite_pending(null);
        }
      },
      { defer: true },
    ),
  );

  createEffect(
    on(
      () => [selected_minutes(), fill_from_last()] as const,
      () => {
        const pending = overwrite_pending();
        if (!pending) {
          return;
        }

        const updated = label_overwrite_pending_upd_get(
          pending,
          {
            selected_minutes: selected_minutes(),
            fill_from_last: fill_from_last(),
            last_label_end_ts: last_label()?.end_ts ?? null,
            extend_fwd: extend_fwd(),
          },
          new Date(),
        );

        set_overwrite_pending(updated);
      },
      { defer: true },
    ),
  );

  function extend_fwd_toggle() {
    const next = !extend_fwd();
    set_extend_fwd(next);
    try {
      localStorage.setItem(EXTEND_FWD_KEY, String(next));
    } catch {}
  }

  async function label_now(label: string) {
    const mins = selected_minutes();
    const now = new Date();
    let start: Date;
    let force_extend_fwd = false;

    const last_end_ts = last_label()?.end_ts;
    if (fill_from_last() && last_end_ts) {
      start = iso_date_parse(last_end_ts);
    } else if (mins === 0) {
      start = now;
      force_extend_fwd = true;
    } else {
      start = new Date(now.getTime() - mins * 60_000);
    }
    const effective_extend = force_extend_fwd || extend_fwd();
    try {
      await label_create({
        start_ts: start.toISOString(),
        end_ts: now.toISOString(),
        label,
        confidence: conf_percent() / 100,
        extend_forward: effective_extend,
      });
      set_flash(label);
      set_label_version((v) => v + 1);
      setTimeout(() => set_flash(null), 1500);
    } catch (err: unknown) {
      const msg: string = err instanceof Error ? err.message : "";
      const json_match = msg.match(/\{[\s\S]*\}/);
      if (json_match) {
        try {
          const parsed = JSON.parse(json_match[0]);
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
            set_overwrite_pending({
              label,
              start: start.toISOString(),
              end: now.toISOString(),
              conflicts: spans,
              confidence: conf_percent() / 100,
              extend_forward: effective_extend,
            });
            return;
          }
        } catch {
          /* fall through to generic error */
        }
      }
      set_flash(`Error: ${msg}`);
      setTimeout(() => set_flash(null), 3000);
    }
  }

  async function overwrite_confirm() {
    const pending = overwrite_pending();
    if (!pending) {
      return;
    }
    set_overwrite_pending(null);
    try {
      await label_create({
        start_ts: pending.start,
        end_ts: pending.end,
        label: pending.label,
        confidence: pending.confidence,
        extend_forward: pending.extend_forward,
        overwrite: true,
      });
      set_flash(pending.label);
      set_label_version((v) => v + 1);
      setTimeout(() => set_flash(null), 1500);
    } catch (err: unknown) {
      set_flash(`Error: ${err instanceof Error ? err.message : "overwrite failed"}`);
      setTimeout(() => set_flash(null), 3000);
    }
  }

  async function keep_all_confirm() {
    const pending = overwrite_pending();
    if (!pending) {
      return;
    }
    set_overwrite_pending(null);
    try {
      await label_create({
        start_ts: pending.start,
        end_ts: pending.end,
        label: pending.label,
        confidence: pending.confidence,
        extend_forward: pending.extend_forward,
        allow_overlap: true,
      });
      set_flash(pending.label);
      set_label_version((v) => v + 1);
      setTimeout(() => set_flash(null), 1500);
    } catch (err: unknown) {
      set_flash(`Error: ${err instanceof Error ? err.message : "keep all failed"}`);
      setTimeout(() => set_flash(null), 3000);
    }
  }

  return (
    <div
      style={{
        padding: "8px",
        "border-top": "1px solid var(--border)",
        ...(props.max_height != null
          ? { "max-height": `${props.max_height}px`, "overflow-y": "auto" }
          : {}),
      }}
    >
      <LabelTimePicker
        selected_minutes={selected_minutes}
        set_selected_minutes={set_selected_minutes}
        fill_from_last={fill_from_last}
        set_fill_from_last={set_fill_from_last}
        last_label={last_label}
      />

      <ActivitySummary minutes={selected_minutes} prediction={props.prediction} />

      <Show when={selected_minutes() !== 0 || fill_from_last()}>
        <LabelExtendToggle checked={extend_fwd} on_toggle={extend_fwd_toggle} />
      </Show>

      <LabelConfidence value={conf_percent} on_change={set_conf_percent} />

      <Show when={overwrite_pending()}>
        {(pending) => (
          <LabelOverwrite
            pending={pending()}
            on_confirm={overwrite_confirm}
            on_keep_all={keep_all_confirm}
            on_cancel={() => set_overwrite_pending(null)}
          />
        )}
      </Show>

      <Show when={flash()}>
        <button
          type="button"
          onClick={() => set_flash(null)}
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
              onClick={() => label_now(lbl)}
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

      <LabelLast last_label={last_label} />
    </div>
  );
};
