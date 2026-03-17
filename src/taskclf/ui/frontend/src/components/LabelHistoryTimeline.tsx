import { type Component, createSignal, For, Show } from "solid-js";
import { duration_fmt, time_fmt } from "../lib/date";
import { LABEL_COLORS } from "../lib/labelColors";
import type { TimelineSegment } from "../lib/labelTimeline";

export const LabelHistoryTimeline: Component<{
  segments: TimelineSegment[];
  on_segment_click?: (seg: TimelineSegment, index: number) => void;
}> = (props) => {
  const [tooltip, set_tooltip] = createSignal<{ text: string; x: number } | null>(null);

  return (
    <div style={{ position: "relative", "margin-top": "3px", "margin-bottom": "4px" }}>
      <div
        style={{
          display: "flex",
          height: "7px",
          "border-radius": "3px",
          overflow: "hidden",
          background: "#1a1a1a",
        }}
      >
        <For each={props.segments}>
          {(seg, idx) => (
            <button
              type="button"
              style={{
                "flex-grow": seg.fraction,
                "flex-basis": "0",
                "min-width": seg.label ? "2px" : seg.fraction > 0.005 ? "1px" : "0",
                background: seg.label
                  ? (LABEL_COLORS[seg.label] ?? "#a0a0a0")
                  : "rgba(255,255,255,0.04)",
                cursor: "pointer",
                transition: "opacity 0.1s, background 0.1s",
                border: "none",
                padding: "0",
                height: "100%",
              }}
              onMouseEnter={(e) => {
                const rect = (
                  e.currentTarget.parentElement as HTMLElement
                ).getBoundingClientRect();
                const x =
                  e.currentTarget.getBoundingClientRect().left
                  - rect.left
                  + e.currentTarget.offsetWidth / 2;
                const dur = duration_fmt(seg.end_ms - seg.start_ms);
                const start = time_fmt(new Date(seg.start_ms));
                const end = time_fmt(new Date(seg.end_ms));
                const prefix = seg.label ?? "Unlabeled";
                set_tooltip({ text: `${prefix}  ${start}\u2013${end}  (${dur})`, x });
                if (!seg.label) {
                  e.currentTarget.style.background = "rgba(255,255,255,0.1)";
                }
              }}
              onMouseLeave={(e) => {
                set_tooltip(null);
                if (!seg.label) {
                  e.currentTarget.style.background = "rgba(255,255,255,0.04)";
                }
              }}
              onClick={() => {
                if (props.on_segment_click) {
                  props.on_segment_click(seg, idx());
                }
              }}
            />
          )}
        </For>
      </div>
      <Show when={tooltip()}>
        {(tip) => (
          <div
            style={{
              position: "absolute",
              top: "-22px",
              left: `${Math.max(0, tip().x)}px`,
              transform: "translateX(-50%)",
              background: "#222",
              color: "#e0e0e0",
              padding: "2px 6px",
              "border-radius": "4px",
              "font-size": "0.55rem",
              "white-space": "nowrap",
              "pointer-events": "none",
              "z-index": "10",
              border: "1px solid #333",
            }}
          >
            {tip().text}
          </div>
        )}
      </Show>
    </div>
  );
};
