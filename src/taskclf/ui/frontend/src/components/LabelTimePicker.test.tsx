import { fireEvent, render, screen } from "@solidjs/testing-library";
import { createSignal } from "solid-js";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { LabelTimePicker } from "./LabelTimePicker";

async function flush_solid() {
  await Promise.resolve();
  await Promise.resolve();
}

describe("LabelTimePicker", () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });
  afterEach(() => {
    vi.useRealTimers();
  });

  it("updates gap label when wall clock advances", async () => {
    vi.setSystemTime(new Date("2026-04-05T11:00:00.000Z"));
    const [last] = createSignal({
      start_ts: "2026-04-05T10:00:00.000Z",
      end_ts: "2026-04-05T10:00:00.000Z",
      label: "Build",
      extend_forward: false,
    });

    render(() => (
      <LabelTimePicker
        selected_minutes={() => 0}
        set_selected_minutes={() => {}}
        fill_from_last={() => false}
        set_fill_from_last={() => {}}
        has_current_label={() => false}
        last_label={last}
      />
    ));

    expect(screen.getByRole("button", { name: "gap 1h" })).toBeInTheDocument();

    vi.setSystemTime(new Date("2026-04-05T12:00:00.000Z"));
    vi.advanceTimersByTime(30_000);
    await flush_solid();

    expect(screen.getByRole("button", { name: /^gap 2h/ })).toBeInTheDocument();
  });

  it("hides gap while a current label is active", () => {
    vi.setSystemTime(new Date("2026-04-05T10:00:00.000Z"));
    const [last] = createSignal({
      start_ts: "2026-04-05T08:00:00.000Z",
      end_ts: "2026-04-05T09:00:00.000Z",
      label: "Build",
      extend_forward: false,
    });

    render(() => (
      <LabelTimePicker
        selected_minutes={() => 0}
        set_selected_minutes={() => {}}
        fill_from_last={() => false}
        set_fill_from_last={() => {}}
        has_current_label={() => true}
        last_label={last}
      />
    ));

    expect(screen.queryByRole("button", { name: /^gap / })).not.toBeInTheDocument();
  });

  it("selects fill-from-last when gap button is clicked", () => {
    vi.setSystemTime(new Date("2026-04-05T11:00:00.000Z"));
    const [last] = createSignal({
      start_ts: "2026-04-05T08:00:00.000Z",
      end_ts: "2026-04-05T09:30:00.000Z",
      label: "Build",
      extend_forward: false,
    });
    const set_fill = vi.fn();

    render(() => (
      <LabelTimePicker
        selected_minutes={() => 5}
        set_selected_minutes={() => {}}
        fill_from_last={() => false}
        set_fill_from_last={set_fill}
        has_current_label={() => false}
        last_label={last}
      />
    ));

    fireEvent.click(screen.getByRole("button", { name: "gap 1h30m" }));
    expect(set_fill).toHaveBeenCalledWith(true);
  });
});
