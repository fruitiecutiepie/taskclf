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
  createLabel,
  deleteLabel,
  fetchCoreLabels,
  fetchLabelsByDate,
  updateLabel,
} from "../lib/api";
import { fmtDateLabel, shiftDate, todayDateStr } from "../lib/date";
import {
  buildDayTimeline,
  type GapItem,
  itemKey,
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
  const [selectedDate, setSelectedDate] = createSignal(todayDateStr());
  let dateInputRef: HTMLInputElement | undefined;

  const [labels, { refetch }] = createResource(
    () => (props.visible() ? selectedDate() : null),
    async (dateStr) => {
      if (!dateStr) return [];
      return fetchLabelsByDate(dateStr);
    },
  );
  const [coreLabels] = createResource(fetchCoreLabels);

  const [expandedKey, setExpandedKey] = createSignal<string | null>(null);
  const [busy, setBusy] = createSignal(false);
  const [flash, setFlash] = createSignal<string | null>(null);

  createEffect(
    on(
      () => props.latestPrediction?.(),
      () => {
        if (props.visible()) refetch();
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
    return buildDayTimeline(entries, selectedDate());
  });

  function toggleRow(item: TimelineItem) {
    const key = itemKey(item);
    setExpandedKey(expandedKey() === key ? null : key);
    setFlash(null);
  }

  async function handleUpdate(
    item: LabelItem,
    newLabel: string,
    newStart: string,
    newEnd: string,
  ) {
    setBusy(true);
    setFlash(null);
    try {
      await updateLabel({
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

  async function handleDelete(item: LabelItem) {
    setBusy(true);
    setFlash(null);
    try {
      await deleteLabel({
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

  async function handleGapCreate(startTs: string, endTs: string, label: string) {
    setBusy(true);
    setFlash(null);
    try {
      await createLabel({
        start_ts: startTs,
        end_ts: endTs,
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

  const isFutureDate = () => selectedDate() >= shiftDate(todayDateStr(), 1);

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
          onClick={() => setSelectedDate(shiftDate(selectedDate(), -1))}
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
            {fmtDateLabel(selectedDate())}
          </button>
          <input
            ref={dateInputRef}
            type="date"
            value={selectedDate()}
            max={todayDateStr()}
            onInput={(e) => {
              const v = e.currentTarget.value;
              if (v) setSelectedDate(v);
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
            if (!isFutureDate()) setSelectedDate(shiftDate(selectedDate(), 1));
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
            if (!isFutureDate()) e.currentTarget.style.color = "#e0e0e0";
          }}
          onMouseLeave={(e) => {
            if (!isFutureDate()) e.currentTarget.style.color = "#999";
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
            if (!item) return;
            const key = itemKey(item);
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
                  expanded={expandedKey() === itemKey(item)}
                  onToggle={() => toggleRow(item)}
                  onCreate={handleGapCreate}
                  coreLabels={coreLabels() ?? []}
                  busy={busy()}
                  flash={expandedKey() === itemKey(item) ? flash() : null}
                />
              }
            >
              <LabelHistoryRow
                lbl={item as LabelItem}
                dateStr={selectedDate()}
                expanded={expandedKey() === itemKey(item)}
                onToggle={() => toggleRow(item)}
                onUpdate={(newLabel, newStart, newEnd) =>
                  handleUpdate(item as LabelItem, newLabel, newStart, newEnd)
                }
                onDelete={() => handleDelete(item as LabelItem)}
                coreLabels={coreLabels() ?? []}
                busy={busy()}
                flash={expandedKey() === itemKey(item) ? flash() : null}
              />
            </Show>
          )}
        </For>
      </Show>
    </div>
  );
};
