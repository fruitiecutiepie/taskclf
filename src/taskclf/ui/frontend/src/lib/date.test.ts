import { describe, expect, it } from "vitest";
import { gap_shortcut_label_from_end } from "./date";

describe("gap_shortcut_label_from_end", () => {
  it("returns null when under one rounded minute", () => {
    const end = Date.parse("2026-04-05T10:00:00.000Z");
    const now = end + 29_000;
    expect(gap_shortcut_label_from_end(end, now)).toBeNull();
  });

  it("formats minutes and hours like the gap button", () => {
    const end = Date.parse("2026-04-05T09:00:00.000Z");
    const now1 = Date.parse("2026-04-05T10:00:00.000Z");
    expect(gap_shortcut_label_from_end(end, now1)).toBe("gap 1h");

    const now2 = Date.parse("2026-04-05T10:30:00.000Z");
    expect(gap_shortcut_label_from_end(end, now2)).toBe("gap 1h30m");
  });
});
