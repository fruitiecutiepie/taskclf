import { describe, expect, it } from "vitest";

import { suggestion_banner_ttl_ms_from_seconds } from "./ws";

describe("suggestion_banner_ttl_ms_from_seconds", () => {
  it("returns null when disabled or invalid", () => {
    expect(suggestion_banner_ttl_ms_from_seconds(0)).toBeNull();
    expect(suggestion_banner_ttl_ms_from_seconds(-1)).toBeNull();
    expect(suggestion_banner_ttl_ms_from_seconds(Number.NaN)).toBeNull();
    expect(suggestion_banner_ttl_ms_from_seconds(Number.POSITIVE_INFINITY)).toBeNull();
  });

  it("returns milliseconds for positive seconds", () => {
    expect(suggestion_banner_ttl_ms_from_seconds(1)).toBe(1000);
    expect(suggestion_banner_ttl_ms_from_seconds(600)).toBe(600_000);
    expect(suggestion_banner_ttl_ms_from_seconds(1.7)).toBe(1000);
  });
});
