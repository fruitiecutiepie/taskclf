import { fireEvent, render, screen, waitFor } from "@solidjs/testing-library";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { core_labels_list, label_update, labels_list } from "../lib/api";
import { LabelRecorder } from "./LabelRecorder";

vi.mock("../lib/api", () => ({
  core_labels_list: vi.fn(),
  labels_list: vi.fn(),
  label_create: vi.fn(),
  label_update: vi.fn(),
}));

vi.mock("./ActivitySummary", () => ({
  ActivitySummary: () => <div data-testid="activity-summary" />,
}));

vi.mock("./PredictionSuggestion", () => ({
  PredictionSuggestion: () => null,
}));

beforeEach(() => {
  vi.clearAllMocks();
  vi.useRealTimers();
  vi.mocked(core_labels_list).mockResolvedValue(["Build", "Write"]);
});

describe("LabelRecorder", () => {
  it("shows a stop action for the current open-ended label and ends it at click time", async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date("2026-04-05T10:00:00Z"));
    vi.mocked(labels_list)
      .mockResolvedValueOnce([
        {
          start_ts: "2026-04-05T09:00:00Z",
          end_ts: "2026-04-05T09:00:00Z",
          label: "Build",
          provenance: "manual",
          user_id: null,
          confidence: 1,
          extend_forward: true,
        },
      ])
      .mockResolvedValueOnce([
        {
          start_ts: "2026-04-05T09:00:00Z",
          end_ts: "2026-04-05T10:00:00.000Z",
          label: "Build",
          provenance: "manual",
          user_id: null,
          confidence: 1,
          extend_forward: false,
        },
      ]);
    vi.mocked(label_update).mockResolvedValue({
      start_ts: "2026-04-05T09:00:00Z",
      end_ts: "2026-04-05T10:00:00.000Z",
      label: "Build",
      provenance: "manual",
      user_id: null,
      confidence: 1,
      extend_forward: false,
    });

    render(() => <LabelRecorder on_collapse={vi.fn()} />);

    expect(await screen.findByText(/^Current:/)).toBeInTheDocument();

    fireEvent.click(await screen.findByRole("button", { name: "Stop current label" }));
    expect(
      screen.getByText("Stop recording the current label now?"),
    ).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Confirm stop" }));

    await waitFor(() => {
      expect(label_update).toHaveBeenCalledWith({
        start_ts: "2026-04-05T09:00:00Z",
        end_ts: "2026-04-05T09:00:00Z",
        label: "Build",
        new_end_ts: "2026-04-05T10:00:00.000Z",
        extend_forward: false,
      });
    });

    await waitFor(() => {
      expect(
        screen.queryByRole("button", { name: "Stop current label" }),
      ).not.toBeInTheDocument();
    });
    await waitFor(() => {
      expect(
        screen.getByText(
          (_content, element) => element?.textContent?.startsWith("Last:") ?? false,
        ),
      ).toBeInTheDocument();
    });
    vi.useRealTimers();
  });

  it("does not show the stop action for completed labels", async () => {
    vi.mocked(labels_list).mockResolvedValue([
      {
        start_ts: "2026-04-05T09:00:00Z",
        end_ts: "2026-04-05T09:30:00Z",
        label: "Build",
        provenance: "manual",
        user_id: null,
        confidence: 1,
        extend_forward: false,
      },
    ]);

    render(() => <LabelRecorder on_collapse={vi.fn()} />);

    await waitFor(() => {
      expect(labels_list).toHaveBeenCalledWith(1);
    });
    await waitFor(() => {
      expect(
        screen.getByText(
          (_content, element) => element?.textContent?.startsWith("Last:") ?? false,
        ),
      ).toBeInTheDocument();
    });
    expect(
      screen.queryByRole("button", { name: "Stop current label" }),
    ).not.toBeInTheDocument();
  });
});
