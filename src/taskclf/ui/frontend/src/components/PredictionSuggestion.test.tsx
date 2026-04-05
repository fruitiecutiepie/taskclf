import { fireEvent, render, screen, waitFor } from "@solidjs/testing-library";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { aw_live_list, feature_summary_get, notification_accept } from "../lib/api";
import { time_format } from "../lib/format";
import type { LabelSuggestion } from "../lib/ws";
import { PredictionSuggestion } from "./PredictionSuggestion";

vi.mock("../lib/api", () => ({
  notification_accept: vi.fn(),
  notification_skip: vi.fn(),
  aw_live_list: vi.fn().mockResolvedValue([]),
  feature_summary_get: vi.fn().mockResolvedValue({
    top_apps: [],
    mean_keys_per_min: null,
    mean_clicks_per_min: null,
    mean_scroll_per_min: null,
    total_buckets: 0,
    session_count: 0,
  }),
}));

vi.mock("../lib/log", () => ({
  frontend_log_error: vi.fn(),
}));

const clipboard_write = vi.fn().mockResolvedValue(undefined);

beforeEach(() => {
  vi.clearAllMocks();
  Object.defineProperty(window.navigator, "clipboard", {
    configurable: true,
    value: { writeText: clipboard_write },
  });
});

function suggestion_make(overrides: Partial<LabelSuggestion> = {}): LabelSuggestion {
  return {
    type: "suggest_label",
    reason: "App transition suggested a new label",
    old_label: "ReadResearch",
    suggested: "Write",
    confidence: 0.92,
    block_start: "2026-04-05T12:00:00Z",
    block_end: "2026-04-05T13:00:00Z",
    ...overrides,
  };
}

function suggestion_range_text(block_start: string, block_end: string): string {
  const start = new Date(block_start);
  const end = new Date(block_end);
  const crosses_local_day =
    start.getFullYear() !== end.getFullYear()
    || start.getMonth() !== end.getMonth()
    || start.getDate() !== end.getDate();

  if (!crosses_local_day) {
    return `${time_format(block_start)} → ${time_format(block_end)}`;
  }

  return `${start.toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  })} → ${end.toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  })}`;
}

describe("PredictionSuggestion", () => {
  it("shows the applicable suggestion time range", () => {
    const suggestion = suggestion_make();

    render(() => <PredictionSuggestion suggestion={() => suggestion} />);

    expect(
      screen.getByText(
        suggestion_range_text(suggestion.block_start, suggestion.block_end),
      ),
    ).toBeInTheDocument();
  });

  it("loads activity context for the suggestion block range", async () => {
    const suggestion = suggestion_make();

    render(() => <PredictionSuggestion suggestion={() => suggestion} />);

    await waitFor(() => {
      expect(vi.mocked(feature_summary_get)).toHaveBeenCalledWith(
        suggestion.block_start,
        suggestion.block_end,
      );
    });
    expect(vi.mocked(aw_live_list)).toHaveBeenCalledWith(
      suggestion.block_start,
      suggestion.block_end,
    );
    expect(
      await screen.findByText("No activity data for this window"),
    ).toBeInTheDocument();
  });

  it("includes dates when the suggestion spans local midnight", () => {
    const suggestion = suggestion_make({
      block_start: new Date(2026, 3, 5, 23, 50, 0).toISOString(),
      block_end: new Date(2026, 3, 6, 0, 10, 0).toISOString(),
    });

    render(() => <PredictionSuggestion suggestion={() => suggestion} />);

    expect(
      screen.getByText(
        suggestion_range_text(suggestion.block_start, suggestion.block_end),
      ),
    ).toBeInTheDocument();
  });

  it("saves immediately when Use suggestion is clicked", async () => {
    const suggestion = suggestion_make();
    const on_saved = vi.fn();
    const on_dismiss = vi.fn();

    render(() => (
      <PredictionSuggestion
        suggestion={() => suggestion}
        on_saved={on_saved}
        on_dismiss={on_dismiss}
      />
    ));

    fireEvent.click(screen.getByRole("button", { name: "Use suggestion" }));

    await waitFor(() => {
      expect(vi.mocked(notification_accept)).toHaveBeenCalledWith({
        block_start: suggestion.block_start,
        block_end: suggestion.block_end,
        label: suggestion.suggested,
      });
    });
    expect(on_saved).toHaveBeenCalledOnce();
    expect(on_dismiss).toHaveBeenCalledOnce();
  });

  it("keeps save errors visible until closed and allows copying them", async () => {
    vi.mocked(notification_accept).mockRejectedValueOnce(new Error("save failed"));
    const suggestion = suggestion_make();

    render(() => <PredictionSuggestion suggestion={() => suggestion} />);

    fireEvent.click(screen.getByRole("button", { name: "Use suggestion" }));

    expect(await screen.findByRole("alert")).toHaveTextContent("save failed");

    fireEvent.click(screen.getByRole("button", { name: "Copy error" }));
    await waitFor(() => expect(clipboard_write).toHaveBeenCalledWith("save failed"));

    fireEvent.click(screen.getByRole("button", { name: "Close error" }));
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });
});
