import { describe, it, expect } from "vitest";
import { labelOverwritePendingUpdGet, type TimeSelection } from "./labelOverwritePendingUpdGet";
import type { OverwritePending } from "../components/LabelOverwrite";

function iso(h: number, m: number): string {
  return `2026-03-14T${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}:00+00:00`;
}

function makeDate(h: number, m: number): Date {
  return new Date(`2026-03-14T${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}:00+00:00`);
}

function basePending(overrides: Partial<OverwritePending> = {}): OverwritePending {
  return {
    label: "Write",
    start: iso(11, 1),
    end: iso(11, 6),
    conflicts: [
      { start_ts: iso(11, 1), end_ts: iso(11, 2), label: "Communicate" },
    ],
    confidence: 1,
    extendForward: false,
    ...overrides,
  };
}

function baseSel(overrides: Partial<TimeSelection> = {}): TimeSelection {
  return {
    selectedMinutes: 5,
    fillFromLast: false,
    lastLabelEndTs: null,
    extendFwd: true,
    ...overrides,
  };
}

describe("labelOverwritePendingUpdGet", () => {
  // ------------------------------------------------------------------
  // User's exact scenario: overlap disappears after time passes
  // ------------------------------------------------------------------
  it("returns null when time has passed and conflicts no longer overlap", () => {
    // Communicate 11:01–11:02, originally clicked at 11:06 with 5m → overlaps.
    // Now it's 11:08, user picks 5m again → start=11:03, no overlap.
    const pending = basePending();
    const now = makeDate(11, 8);

    const result = labelOverwritePendingUpdGet(pending, baseSel({ selectedMinutes: 5 }), now);
    expect(result).toBeNull();
  });

  it("preserves conflicts that still overlap after time change", () => {
    // Communicate 11:01–11:02. Now it's 11:08, user picks 15m → start=10:53.
    // Still overlaps.
    const pending = basePending();
    const now = makeDate(11, 8);

    const result = labelOverwritePendingUpdGet(pending, baseSel({ selectedMinutes: 15 }), now);
    expect(result).not.toBeNull();
    expect(result!.conflicts).toHaveLength(1);
    expect(result!.conflicts[0].label).toBe("Communicate");
  });

  // ------------------------------------------------------------------
  // Filters out non-overlapping conflicts from a multi-conflict set
  // ------------------------------------------------------------------
  it("filters out conflicts that no longer overlap", () => {
    const pending = basePending({
      conflicts: [
        { start_ts: iso(10, 50), end_ts: iso(10, 55), label: "Debug" },
        { start_ts: iso(11, 1), end_ts: iso(11, 2), label: "Communicate" },
        { start_ts: iso(11, 5), end_ts: iso(11, 10), label: "Review" },
      ],
    });
    // Now 11:08, user picks 5m → start=11:03, end=11:08.
    // Only Review (11:05–11:10) overlaps [11:03, 11:08).
    const now = makeDate(11, 8);
    const result = labelOverwritePendingUpdGet(pending, baseSel({ selectedMinutes: 5 }), now);
    expect(result).not.toBeNull();
    expect(result!.conflicts).toHaveLength(1);
    expect(result!.conflicts[0].label).toBe("Review");
  });

  // ------------------------------------------------------------------
  // Updates start and end timestamps
  // ------------------------------------------------------------------
  it("updates start and end to reflect current time", () => {
    const pending = basePending({
      conflicts: [
        { start_ts: iso(10, 0), end_ts: iso(12, 0), label: "Build" },
      ],
    });
    const now = makeDate(11, 30);
    const result = labelOverwritePendingUpdGet(pending, baseSel({ selectedMinutes: 10 }), now);
    expect(result).not.toBeNull();
    expect(new Date(result!.start).getUTCMinutes()).toBe(20); // 11:20
    expect(new Date(result!.end).getUTCMinutes()).toBe(30);   // 11:30
  });

  // ------------------------------------------------------------------
  // fillFromLast uses lastLabelEndTs as start
  // ------------------------------------------------------------------
  it("uses lastLabelEndTs as start when fillFromLast is true", () => {
    const pending = basePending({
      conflicts: [
        { start_ts: iso(10, 0), end_ts: iso(12, 0), label: "Build" },
      ],
    });
    const now = makeDate(11, 30);
    const result = labelOverwritePendingUpdGet(
      pending,
      baseSel({ fillFromLast: true, lastLabelEndTs: iso(10, 45) }),
      now,
    );
    expect(result).not.toBeNull();
    expect(new Date(result!.start).getUTCMinutes()).toBe(45); // 10:45
  });

  it("returns null when fillFromLast range does not overlap any conflict", () => {
    const pending = basePending({
      conflicts: [
        { start_ts: iso(9, 0), end_ts: iso(9, 30), label: "Debug" },
      ],
    });
    const now = makeDate(11, 30);
    const result = labelOverwritePendingUpdGet(
      pending,
      baseSel({ fillFromLast: true, lastLabelEndTs: iso(10, 0) }),
      now,
    );
    expect(result).toBeNull();
  });

  // ------------------------------------------------------------------
  // selectedMinutes=0 sets extendForward=true
  // ------------------------------------------------------------------
  it("forces extendForward when selectedMinutes is 0", () => {
    const pending = basePending({
      start: iso(11, 6),
      end: iso(11, 6),
      conflicts: [
        { start_ts: iso(11, 5), end_ts: iso(11, 7), label: "Build" },
      ],
      extendForward: false,
    });
    const now = makeDate(11, 6);
    const result = labelOverwritePendingUpdGet(
      pending,
      baseSel({ selectedMinutes: 0, extendFwd: false }),
      now,
    );
    expect(result).not.toBeNull();
    expect(result!.extendForward).toBe(true);
  });

  it("uses extendFwd from selection when selectedMinutes > 0", () => {
    const pending = basePending({
      conflicts: [
        { start_ts: iso(10, 0), end_ts: iso(12, 0), label: "Build" },
      ],
    });
    const now = makeDate(11, 10);
    const result = labelOverwritePendingUpdGet(
      pending,
      baseSel({ selectedMinutes: 5, extendFwd: false }),
      now,
    );
    expect(result).not.toBeNull();
    expect(result!.extendForward).toBe(false);
  });

  // ------------------------------------------------------------------
  // Preserves label, confidence from original pending
  // ------------------------------------------------------------------
  it("preserves label and confidence from the original pending", () => {
    const pending = basePending({
      label: "Communicate",
      confidence: 0.8,
      conflicts: [
        { start_ts: iso(10, 0), end_ts: iso(12, 0), label: "Build" },
      ],
    });
    const now = makeDate(11, 10);
    const result = labelOverwritePendingUpdGet(pending, baseSel({ selectedMinutes: 5 }), now);
    expect(result).not.toBeNull();
    expect(result!.label).toBe("Communicate");
    expect(result!.confidence).toBe(0.8);
  });
});
