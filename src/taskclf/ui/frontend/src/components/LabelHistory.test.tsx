import { render, screen, waitFor } from "@solidjs/testing-library";
import { createSignal } from "solid-js";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { core_labels_list, labels_list_by_date } from "../lib/api";
import { LabelHistory } from "./LabelHistory";

vi.mock("../lib/api", () => ({
  core_labels_list: vi.fn(),
  label_create: vi.fn(),
  label_delete: vi.fn(),
  label_update: vi.fn(),
  labels_list_by_date: vi.fn(),
}));

describe("LabelHistory", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(core_labels_list).mockResolvedValue(["Build", "Write"]);
  });

  it("refetches the selected day when the label change count increments", async () => {
    vi.mocked(labels_list_by_date)
      .mockResolvedValueOnce([
        {
          start_ts: "2026-04-05T09:00:00Z",
          end_ts: "2026-04-05T10:00:00Z",
          label: "Build",
          provenance: "manual",
          user_id: null,
          confidence: 1,
          extend_forward: false,
        },
      ])
      .mockResolvedValueOnce([
        {
          start_ts: "2026-04-05T09:00:00Z",
          end_ts: "2026-04-05T10:00:00Z",
          label: "Build",
          provenance: "manual",
          user_id: null,
          confidence: 1,
          extend_forward: false,
        },
        {
          start_ts: "2026-04-05T10:00:00Z",
          end_ts: "2026-04-05T11:00:00Z",
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
});
