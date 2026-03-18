import { describe, expect, it } from "vitest";
import { day_timeline_build } from "./labelTimeline";

function iso(h: number, m: number): string {
  return `2026-03-14T${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}:00+00:00`;
}

describe("day_timeline_build", () => {
  it("renders an open-ended now-label as a label item", () => {
    const start = iso(11, 6);
    const result = day_timeline_build(
      [
        {
          label: "Write",
          start_ts: start,
          end_ts: start,
          extend_forward: true,
        },
      ],
      "2026-03-14",
    );

    const label_item = result.items.find((item) => item.kind === "label");
    expect(label_item).toBeDefined();
    if (label_item?.kind !== "label") {
      throw new Error("expected label item");
    }
    expect(label_item.label).toBe("Write");
    expect(label_item.open_ended).toBe(true);
    expect(result.segments.some((seg) => seg.label === "Write")).toBe(true);
  });

  it("keeps zero-length non-extend labels hidden", () => {
    const ts = iso(11, 6);
    const result = day_timeline_build(
      [{ label: "Write", start_ts: ts, end_ts: ts, extend_forward: false }],
      "2026-03-14",
    );

    expect(result.items.some((item) => item.kind === "label")).toBe(false);
    expect(result.segments.some((seg) => seg.label === "Write")).toBe(false);
  });
});
