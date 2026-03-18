import {
  type Accessor,
  type Component,
  createEffect,
  createMemo,
  createResource,
  createSignal,
  For,
  on,
  Show,
} from "solid-js";
import {
  core_labels_list,
  label_create,
  label_delete,
  label_update,
  labels_list_by_date,
} from "../lib/api";
import { date_label_fmt, date_shift, date_today_str } from "../lib/date";
import {
  day_timeline_build,
  type GapItem,
  item_key,
  type LabelEntry,
  type LabelItem,
  type TimelineItem,
} from "../lib/labelTimeline";
import type { Prediction } from "../lib/ws";
import { LabelHistoryGapRow } from "./LabelHistoryGapRow";
import { LabelHistoryRow } from "./LabelHistoryRow";
import { LabelHistoryTimeline } from "./LabelHistoryTimeline";

export const LabelHistory: Component<{
  visible: Accessor<boolean>;
  latest_prediction?: Accessor<Prediction | null>;
}> = (props) => {
  const [selected_date, set_selected_date] = createSignal(date_today_str());
  let date_input_ref: HTMLInputElement | undefined;

  const [labels, { refetch }] = createResource(
    () => (props.visible() ? selected_date() : null),
    async (dateStr) => {
      if (!dateStr) {
        return [];
      }
      return labels_list_by_date(dateStr);
    },
  );
  const [coreLabels] = createResource(core_labels_list);

  const [expanded_key, set_expanded_key] = createSignal<string | null>(null);
  const [busy, set_busy] = createSignal(false);
  const [flash, set_flash] = createSignal<string | null>(null);

  createEffect(
    on(
      () => props.latest_prediction?.(),
      () => {
        if (props.visible()) {
          refetch();
        }
      },
      { defer: true },
    ),
  );

  const day_data = createMemo(() => {
    const l = labels();
    const entries: LabelEntry[] = (l ?? []).map((r) => ({
      label: r.label,
      start_ts: r.start_ts,
      end_ts: r.end_ts,
      extend_forward: r.extend_forward,
    }));
    return day_timeline_build(entries, selected_date());
  });

  function row_toggle(item: TimelineItem) {
    const key = item_key(item);
    set_expanded_key(expanded_key() === key ? null : key);
    set_flash(null);
  }

  async function label_update_submit(
    item: LabelItem,
    new_label: string,
    new_start: string,
    new_end: string,
  ) {
    set_busy(true);
    set_flash(null);
    try {
      await label_update({
        start_ts: item.start_ts,
        end_ts: item.end_ts,
        label: new_label,
        new_start_ts: new_start,
        new_end_ts: new_end,
      });
      set_flash(new_label);
      setTimeout(() => {
        set_flash(null);
        set_expanded_key(null);
        refetch();
      }, 800);
    } catch (err: unknown) {
      set_flash(`Error: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      set_busy(false);
    }
  }

  async function label_delete_submit(item: LabelItem) {
    set_busy(true);
    set_flash(null);
    try {
      await label_delete({
        start_ts: item.start_ts,
        end_ts: item.end_ts,
      });
      set_expanded_key(null);
      refetch();
    } catch (err: unknown) {
      set_flash(`Error: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      set_busy(false);
    }
  }

  async function gap_create_submit(start_ts: string, end_ts: string, label: string) {
    set_busy(true);
    set_flash(null);
    try {
      await label_create({
        start_ts,
        end_ts,
        label,
      });
      set_flash(label);
      setTimeout(() => {
        set_flash(null);
        set_expanded_key(null);
        refetch();
      }, 800);
    } catch (err: unknown) {
      set_flash(`Error: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      set_busy(false);
    }
  }

  const is_future_date = () => selected_date() >= date_shift(date_today_str(), 1);

  return (
    <div
      style={{
        padding: "6px 8px",
        "font-family": "'SF Mono', 'Fira Code', 'Cascadia Code', monospace",
        "font-size": "0.65rem",
        color: "#e0e0e0",
      }}
    >
      <div
        style={{
          display: "flex",
          "align-items": "center",
          "justify-content": "space-between",
          "margin-bottom": "6px",
          "padding-bottom": "4px",
          "border-bottom": "1px solid #2a2a2a",
        }}
      >
        <button
          type="button"
          onClick={() => set_selected_date(date_shift(selected_date(), -1))}
          style={{
            background: "none",
            border: "none",
            color: "#999",
            cursor: "pointer",
            "font-size": "0.75rem",
            padding: "2px 6px",
            "border-radius": "4px",
            transition: "color 0.1s",
          }}
          onMouseEnter={(e) => {
            e.currentTarget.style.color = "#e0e0e0";
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.color = "#999";
          }}
        >
          ◀
        </button>
        <div style={{ position: "relative" }}>
          <button
            type="button"
            onClick={() => date_input_ref?.showPicker()}
            style={{
              "font-size": "0.75rem",
              "font-weight": "700",
              color: "#e0e0e0",
              "letter-spacing": "0.02em",
              cursor: "pointer",
              "user-select": "none",
              background: "none",
              border: "none",
              padding: "0",
              font: "inherit",
            }}
          >
            {date_label_fmt(selected_date())}
          </button>
          <input
            ref={date_input_ref}
            type="date"
            value={selected_date()}
            max={date_today_str()}
            onInput={(e) => {
              const v = e.currentTarget.value;
              if (v) {
                set_selected_date(v);
              }
            }}
            style={{
              position: "absolute",
              top: "0",
              left: "0",
              width: "100%",
              height: "100%",
              opacity: "0",
              cursor: "pointer",
            }}
          />
        </div>
        <button
          type="button"
          onClick={() => {
            if (!is_future_date()) {
              set_selected_date(date_shift(selected_date(), 1));
            }
          }}
          disabled={is_future_date()}
          style={{
            background: "none",
            border: "none",
            color: is_future_date() ? "#444" : "#999",
            cursor: is_future_date() ? "default" : "pointer",
            "font-size": "0.75rem",
            padding: "2px 6px",
            "border-radius": "4px",
            transition: "color 0.1s",
          }}
          onMouseEnter={(e) => {
            if (!is_future_date()) {
              e.currentTarget.style.color = "#e0e0e0";
            }
          }}
          onMouseLeave={(e) => {
            if (!is_future_date()) {
              e.currentTarget.style.color = "#999";
            }
          }}
        >
          ▶
        </button>
      </div>

      <Show
        when={!labels.loading}
        fallback={
          <div
            style={{
              "text-align": "center",
              padding: "16px 8px",
              color: "#707070",
              "font-size": "0.6rem",
            }}
          >
            Loading labels…
          </div>
        }
      >
        <LabelHistoryTimeline
          segments={day_data().segments}
          on_segment_click={(_seg, index) => {
            const item = day_data().items[index];
            if (!item) {
              return;
            }
            const key = item_key(item);
            set_expanded_key(expanded_key() === key ? null : key);
            set_flash(null);
          }}
        />
        <For each={day_data().items}>
          {(item) => (
            <Show
              when={item.kind === "label"}
              fallback={
                <LabelHistoryGapRow
                  gap={item as GapItem}
                  date_str={selected_date()}
                  expanded={expanded_key() === item_key(item)}
                  on_toggle={() => row_toggle(item)}
                  on_create={gap_create_submit}
                  core_labels={coreLabels() ?? []}
                  busy={busy()}
                  flash={expanded_key() === item_key(item) ? flash() : null}
                />
              }
            >
              <LabelHistoryRow
                label_item={item as LabelItem}
                date_str={selected_date()}
                expanded={expanded_key() === item_key(item)}
                on_toggle={() => row_toggle(item)}
                on_update={(new_label, new_start, new_end) =>
                  label_update_submit(item as LabelItem, new_label, new_start, new_end)
                }
                on_delete={() => label_delete_submit(item as LabelItem)}
                core_labels={coreLabels() ?? []}
                busy={busy()}
                flash={expanded_key() === item_key(item) ? flash() : null}
              />
            </Show>
          )}
        </For>
      </Show>
    </div>
  );
};
