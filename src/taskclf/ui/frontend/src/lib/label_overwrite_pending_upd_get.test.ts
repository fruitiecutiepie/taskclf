import { describe, expect, it } from "vitest";
import type { OverwritePending } from "../components/LabelOverwrite";
import {
  label_overwrite_pending_upd_get,
  type TimeSelection,
} from "./label_overwrite_pending_upd_get";

function iso(h: number, m: number): string {
  return `2026-03-14T${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}:00+00:00`;
}

function date_make(h: number, m: number): Date {
  return new Date(
    `2026-03-14T${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}:00+00:00`,
  );
}

function pending_base(overrides: Partial<OverwritePending> = {}): OverwritePending {
  return {
    label: "Write",
    start: iso(11, 1),
    end: iso(11, 6),
    conflicts: [{ start_ts: iso(11, 1), end_ts: iso(11, 2), label: "Communicate" }],
    confidence: 1,
    extend_forward: false,
    ...overrides,
  };
}

function selection_base(overrides: Partial<TimeSelection> = {}): TimeSelection {
  return {
    selected_minutes: 5,
    fill_from_last: false,
    last_label_end_ts: null,
    extend_fwd: true,
    ...overrides,
  };
}

describe("label_overwrite_pending_upd_get", () => {
  it("returns null when time has passed and conflicts no longer overlap", () => {
    const pending = pending_base();
    const now = date_make(11, 8);

    const result = label_overwrite_pending_upd_get(
      pending,
      selection_base({ selected_minutes: 5 }),
      now,
    );
    expect(result).toBeNull();
  });

  it("preserves conflicts that still overlap after time change", () => {
    const pending = pending_base();
    const now = date_make(11, 8);

    const result = label_overwrite_pending_upd_get(
      pending,
      selection_base({ selected_minutes: 15 }),
      now,
    );
    expect(result).not.toBeNull();
    expect(result?.conflicts).toHaveLength(1);
    expect(result?.conflicts[0].label).toBe("Communicate");
  });

  it("filters out conflicts that no longer overlap", () => {
    const pending = pending_base({
      conflicts: [
        { start_ts: iso(10, 50), end_ts: iso(10, 55), label: "Debug" },
        { start_ts: iso(11, 1), end_ts: iso(11, 2), label: "Communicate" },
        { start_ts: iso(11, 5), end_ts: iso(11, 10), label: "Review" },
      ],
    });
    const now = date_make(11, 8);
    const result = label_overwrite_pending_upd_get(
      pending,
      selection_base({ selected_minutes: 5 }),
      now,
    );
    expect(result).not.toBeNull();
    expect(result?.conflicts).toHaveLength(1);
    expect(result?.conflicts[0].label).toBe("Review");
  });

  it("updates start and end to reflect current time", () => {
    const pending = pending_base({
      conflicts: [{ start_ts: iso(10, 0), end_ts: iso(12, 0), label: "Build" }],
    });
    const now = date_make(11, 30);
    const result = label_overwrite_pending_upd_get(
      pending,
      selection_base({ selected_minutes: 10 }),
      now,
    );
    if (result === null) {
      throw new Error("expected non-null result");
    }
    expect(new Date(result.start).getUTCMinutes()).toBe(20);
    expect(new Date(result.end).getUTCMinutes()).toBe(30);
  });

  it("uses lastLabelEndTs as start when fillFromLast is true", () => {
    const pending = pending_base({
      conflicts: [{ start_ts: iso(10, 0), end_ts: iso(12, 0), label: "Build" }],
    });
    const now = date_make(11, 30);
    const result = label_overwrite_pending_upd_get(
      pending,
      selection_base({ fill_from_last: true, last_label_end_ts: iso(10, 45) }),
      now,
    );
    if (result === null) {
      throw new Error("expected non-null result");
    }
    expect(new Date(result.start).getUTCMinutes()).toBe(45);
  });

  it("returns null when fillFromLast range does not overlap any conflict", () => {
    const pending = pending_base({
      conflicts: [{ start_ts: iso(9, 0), end_ts: iso(9, 30), label: "Debug" }],
    });
    const now = date_make(11, 30);
    const result = label_overwrite_pending_upd_get(
      pending,
      selection_base({ fill_from_last: true, last_label_end_ts: iso(10, 0) }),
      now,
    );
    expect(result).toBeNull();
  });

  it("forces extend_forward when selected_minutes is 0", () => {
    const pending = pending_base({
      start: iso(11, 6),
      end: iso(11, 6),
      conflicts: [{ start_ts: iso(11, 5), end_ts: iso(11, 7), label: "Build" }],
      extend_forward: false,
    });
    const now = date_make(11, 6);
    const result = label_overwrite_pending_upd_get(
      pending,
      selection_base({ selected_minutes: 0, extend_fwd: false }),
      now,
    );
    expect(result).not.toBeNull();
    expect(result?.extend_forward).toBe(true);
  });

  it("uses extend_fwd from selection when selected_minutes > 0", () => {
    const pending = pending_base({
      conflicts: [{ start_ts: iso(10, 0), end_ts: iso(12, 0), label: "Build" }],
    });
    const now = date_make(11, 10);
    const result = label_overwrite_pending_upd_get(
      pending,
      selection_base({ selected_minutes: 5, extend_fwd: false }),
      now,
    );
    expect(result).not.toBeNull();
    expect(result?.extend_forward).toBe(false);
  });

  it("preserves label and confidence from the original pending", () => {
    const pending = pending_base({
      label: "Communicate",
      confidence: 0.8,
      conflicts: [{ start_ts: iso(10, 0), end_ts: iso(12, 0), label: "Build" }],
    });
    const now = date_make(11, 10);
    const result = label_overwrite_pending_upd_get(
      pending,
      selection_base({ selected_minutes: 5 }),
      now,
    );
    expect(result).not.toBeNull();
    expect(result?.label).toBe("Communicate");
    expect(result?.confidence).toBe(0.8);
  });
});
