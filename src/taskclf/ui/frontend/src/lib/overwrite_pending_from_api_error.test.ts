import { describe, expect, it } from "vitest";
import { overwrite_pending_from_api_error } from "./overwrite_pending_from_api_error";

describe("overwrite_pending_from_api_error", () => {
  const params = {
    label: "Write",
    start: "2026-04-05T12:00:00Z",
    end: "2026-04-05T13:00:00Z",
    confidence: 0.92,
    extend_forward: false,
  };

  it("returns null when the message has no JSON", () => {
    expect(overwrite_pending_from_api_error(new Error("network"), params)).toBeNull();
  });

  it("builds pending from structured detail.conflicting_spans", () => {
    const body = {
      detail: {
        error: "overlap",
        conflicting_spans: [
          {
            start_ts: "2026-04-05T12:00:00+00:00",
            end_ts: "2026-04-05T13:00:00+00:00",
            label: "Build",
          },
        ],
      },
    };
    const err = new Error(`409: ${JSON.stringify(body)}`);
    const p = overwrite_pending_from_api_error(err, params);
    expect(p).not.toBeNull();
    expect(p?.conflicts).toHaveLength(1);
    expect(p?.conflicts[0].label).toBe("Build");
    expect(p?.label).toBe("Write");
    expect(p?.extend_forward).toBe(false);
  });

  it("builds pending from legacy conflicting_start_ts fields", () => {
    const body = {
      detail: {
        error: "overlap",
        conflicting_start_ts: "2026-04-05T12:00:00+00:00",
        conflicting_end_ts: "2026-04-05T13:00:00+00:00",
        conflicting_label: "Meet",
      },
    };
    const err = new Error(`409: ${JSON.stringify(body)}`);
    const p = overwrite_pending_from_api_error(err, params);
    expect(p?.conflicts).toHaveLength(1);
    expect(p?.conflicts[0].label).toBe("Meet");
  });
});
