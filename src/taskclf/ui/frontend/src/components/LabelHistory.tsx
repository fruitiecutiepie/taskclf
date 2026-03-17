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
  latestPrediction?: Accessor<Prediction | null>;
}> = (props) => {
  const [selectedDate, setSelectedDate] = createSignal(date_today_str());
  let dateInputRef: HTMLInputElement | undefined;

  const [labels, { refetch }] = createResource(
    () => (props.visible() ? selectedDate() : null),
    async (dateStr) => {
      if (!dateStr) {
        return [];
      }
      return labels_list_by_date(dateStr);
    },
  );
  const [coreLabels] = createResource(core_labels_list);

  const [expandedKey, setExpandedKey] = createSignal<string | null>(null);
  const [busy, setBusy] = createSignal(false);
  const [flash, setFlash] = createSignal<string | null>(null);

  createEffect(
    on(
      () => props.latestPrediction?.(),
      () => {
        if (props.visible()) {
          refetch();
        }
      },
      { defer: true },
    ),
  );

  const dayData = createMemo(() => {
    const l = labels();
    const entries: LabelEntry[] = (l ?? []).map((r) => ({
      label: r.label,
      start_ts: r.start_ts,
      end_ts: r.end_ts,
    }));
    return day_timeline_build(entries, selectedDate());
  });

  function row_toggle(item: TimelineItem) {
    const key = item_key(item);
    setExpandedKey(expandedKey() === key ? null : key);
    setFlash(null);
  }

  async function label_update_submit(
    item: LabelItem,
    newLabel: string,
    newStart: string,
    newEnd: string,
  ) {
    setBusy(true);
    setFlash(null);
    try {
      await label_update({
        start_ts: item.start_ts,
        end_ts: item.end_ts,
        label: newLabel,
        new_start_ts: newStart,
        new_end_ts: newEnd,
      });
      setFlash(newLabel);
      setTimeout(() => {
        setFlash(null);
        setExpandedKey(null);
        refetch();
      }, 800);
    } catch (err: unknown) {
      setFlash(`Error: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setBusy(false);
    }
  }

  async function label_delete_submit(item: LabelItem) {
    setBusy(true);
    setFlash(null);
    try {
      await label_delete({
        start_ts: item.start_ts,
        end_ts: item.end_ts,
      });
      setExpandedKey(null);
      refetch();
    } catch (err: unknown) {
      setFlash(`Error: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setBusy(false);
    }
  }

  async function gap_create_submit(start_ts: string, end_ts: string, label: string) {
    setBusy(true);
    setFlash(null);
    try {
      await label_create({
        start_ts,
        end_ts,
        label,
      });
      setFlash(label);
      setTimeout(() => {
        setFlash(null);
        setExpandedKey(null);
        refetch();
      }, 800);
    } catch (err: unknown) {
      setFlash(`Error: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setBusy(false);
    }
  }

  const isFutureDate = () => selectedDate() >= date_shift(date_today_str(), 1);

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
          onClick={() => setSelectedDate(date_shift(selectedDate(), -1))}
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
            onClick={() => dateInputRef?.showPicker()}
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
            {date_label_fmt(selectedDate())}
          </button>
          <input
            ref={dateInputRef}
            type="date"
            value={selectedDate()}
            max={date_today_str()}
            onInput={(e) => {
              const v = e.currentTarget.value;
              if (v) {
                setSelectedDate(v);
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
            if (!isFutureDate()) {
              setSelectedDate(date_shift(selectedDate(), 1));
            }
          }}
          disabled={isFutureDate()}
          style={{
            background: "none",
            border: "none",
            color: isFutureDate() ? "#444" : "#999",
            cursor: isFutureDate() ? "default" : "pointer",
            "font-size": "0.75rem",
            padding: "2px 6px",
            "border-radius": "4px",
            transition: "color 0.1s",
          }}
          onMouseEnter={(e) => {
            if (!isFutureDate()) {
              e.currentTarget.style.color = "#e0e0e0";
            }
          }}
          onMouseLeave={(e) => {
            if (!isFutureDate()) {
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
          segments={dayData().segments}
          onSegmentClick={(_seg, index) => {
            const item = dayData().items[index];
            if (!item) {
              return;
            }
            const key = item_key(item);
            setExpandedKey(expandedKey() === key ? null : key);
            setFlash(null);
          }}
        />
        <For each={dayData().items}>
          {(item) => (
            <Show
              when={item.kind === "label"}
              fallback={
                <LabelHistoryGapRow
                  gap={item as GapItem}
                  dateStr={selectedDate()}
                  expanded={expandedKey() === item_key(item)}
                  onToggle={() => row_toggle(item)}
                  onCreate={gap_create_submit}
                  coreLabels={coreLabels() ?? []}
                  busy={busy()}
                  flash={expandedKey() === item_key(item) ? flash() : null}
                />
              }
            >
              <LabelHistoryRow
                lbl={item as LabelItem}
                dateStr={selectedDate()}
                expanded={expandedKey() === item_key(item)}
                onToggle={() => row_toggle(item)}
                onUpdate={(newLabel, newStart, newEnd) =>
                  label_update_submit(item as LabelItem, newLabel, newStart, newEnd)
                }
                onDelete={() => label_delete_submit(item as LabelItem)}
                coreLabels={coreLabels() ?? []}
                busy={busy()}
                flash={expandedKey() === item_key(item) ? flash() : null}
              />
            </Show>
          )}
        </For>
      </Show>
    </div>
  );
};
