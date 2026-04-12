import { render, screen, waitFor } from "@solidjs/testing-library";
import { createSignal } from "solid-js";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { core_labels_list, labels_list_by_date } from "../lib/api";
import { date_today_str } from "../lib/date";
import { LabelHistory } from "./LabelHistory";

vi.mock("../lib/api", () => ({
  core_labels_list: vi.fn(),
  label_create: vi.fn(),
  label_delete: vi.fn(),
  label_update: vi.fn(),
  labels_list_by_date: vi.fn(),
}));

function iso_at_local_time(date_str: string, hour: number): string {
  const [year, month, day] = date_str.split("-").map(Number);
  return new Date(year, month - 1, day, hour, 0, 0, 0).toISOString();
}

describe("LabelHistory", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(core_labels_list).mockResolvedValue(["Build", "Write"]);
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("refetches the selected day when the label change count increments", async () => {
    const date_str = date_today_str();

    vi.mocked(labels_list_by_date)
      .mockResolvedValueOnce([
        {
          start_ts: iso_at_local_time(date_str, 9),
          end_ts: iso_at_local_time(date_str, 10),
          label: "Build",
          provenance: "manual",
          user_id: null,
          confidence: 1,
          extend_forward: false,
        },
      ])
      .mockResolvedValueOnce([
        {
          start_ts: iso_at_local_time(date_str, 9),
          end_ts: iso_at_local_time(date_str, 10),
          label: "Build",
          provenance: "manual",
          user_id: null,
          confidence: 1,
          extend_forward: false,
        },
        {
          start_ts: iso_at_local_time(date_str, 10),
          end_ts: iso_at_local_time(date_str, 11),
          label: "Write",
          provenance: "suggestion",
          user_id: null,
          confidence: 1,
          extend_forward: false,
        },
      ]);

    const [label_change_count, set_label_change_count] = createSignal(0);

    render(() => (
      <LabelHistory visible={() => true} label_change_count={label_change_count} />
    ));

    expect(await screen.findByText("Build")).toBeInTheDocument();

    set_label_change_count(1);

    expect(await screen.findByText("Write")).toBeInTheDocument();
    await waitFor(() => {
      expect(vi.mocked(labels_list_by_date)).toHaveBeenCalledTimes(2);
    });
  });

  it("resets a stale today view to the current day when history opens", async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date(2026, 3, 5, 10, 0, 0, 0));
    const initial_today = date_today_str();

    vi.mocked(labels_list_by_date).mockResolvedValue([]);

    const [visible, set_visible] = createSignal(false);

    render(() => <LabelHistory visible={visible} />);

    vi.setSystemTime(new Date(2026, 3, 6, 10, 0, 0, 0));
    const next_today = date_today_str();

    set_visible(true);

    await waitFor(() => {
      expect(vi.mocked(labels_list_by_date)).toHaveBeenCalledWith(next_today);
    });
    expect(vi.mocked(labels_list_by_date)).not.toHaveBeenCalledWith(initial_today);
  });
});
