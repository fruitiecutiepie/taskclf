import { fireEvent, render, screen, waitFor } from "@solidjs/testing-library";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { ActivityProviderStatus, ActivitySummary as ActivitySummaryData } from "../lib/api";
import { activity_summary_get, notification_accept } from "../lib/api";
import { time_format } from "../lib/format";
import type { LabelSuggestion } from "../lib/ws";
import { PredictionSuggestion } from "./PredictionSuggestion";

function activity_provider_make(
  overrides: Partial<ActivityProviderStatus> = {},
): ActivityProviderStatus {
  return {
    provider_id: "activitywatch",
    provider_name: "ActivityWatch",
    state: "ready",
    summary_available: true,
    endpoint: "http://localhost:5600",
    source_id: "aw-window-test",
    last_sample_count: 0,
    last_sample_breakdown: {},
    setup_title: "Activity source unavailable",
    setup_message:
      "Manual labeling still works, but activity summaries and automatic activity tracking are unavailable until this source is set up.",
    setup_steps: [
      "Install and start ActivityWatch.",
      "Confirm the local server is reachable at http://localhost:5600.",
      "If you use a custom host, update aw_host in config.toml and restart taskclf.",
    ],
    help_url: "https://activitywatch.net/",
    ...overrides,
  };
}

function activity_summary_make(
  overrides: Partial<ActivitySummaryData> = {},
): ActivitySummaryData {
  const { activity_provider, ...rest } = overrides;
  return {
    activity_provider: activity_provider_make(activity_provider),
    recent_apps: [],
    top_apps: [],
    mean_keys_per_min: null,
    mean_clicks_per_min: null,
    mean_scroll_per_min: null,
    total_buckets: 0,
    session_count: 0,
    range_state: "no_data",
    message: "No activity data for this window",
    ...rest,
  };
}

vi.mock("../lib/api", () => ({
  notification_accept: vi.fn(),
  notification_skip: vi.fn(),
  activity_summary_get: vi.fn().mockResolvedValue(activity_summary_make()),
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
      expect(vi.mocked(activity_summary_get)).toHaveBeenCalledWith(
        suggestion.block_start,
        suggestion.block_end,
      );
    });
    expect(
      await screen.findByText("No activity data for this window"),
    ).toBeInTheDocument();
  });

  it("shows provider setup guidance without blocking suggestion actions", async () => {
    vi.mocked(activity_summary_get).mockResolvedValueOnce(
      activity_summary_make({
        activity_provider: activity_provider_make({
          state: "setup_required",
          summary_available: false,
          source_id: null,
        }),
        range_state: "provider_unavailable",
        message:
          "Manual labeling still works, but activity summaries and automatic activity tracking are unavailable until this source is set up.",
      }),
    );
    const suggestion = suggestion_make();

    render(() => <PredictionSuggestion suggestion={() => suggestion} />);

    expect(await screen.findByText("Activity source unavailable")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Use suggestion" }));

    await waitFor(() => {
      expect(vi.mocked(notification_accept)).toHaveBeenCalledWith({
        block_start: suggestion.block_start,
        block_end: suggestion.block_end,
        label: suggestion.suggested,
      });
    });
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

  it("on overlap 409 shows Overwrite All and retries with overwrite", async () => {
    const suggestion = suggestion_make();
    const on_saved = vi.fn();
    const on_dismiss = vi.fn();
    const body = {
      detail: {
        error: "overlap",
        conflicting_spans: [
          {
            start_ts: "2026-04-05T11:00:00+00:00",
            end_ts: "2026-04-05T14:00:00+00:00",
            label: "Build",
          },
        ],
      },
    };
    vi.mocked(notification_accept)
      .mockRejectedValueOnce(new Error(`409: ${JSON.stringify(body)}`))
      .mockResolvedValueOnce({
        start_ts: suggestion.block_start,
        end_ts: suggestion.block_end,
        label: suggestion.suggested,
        provenance: "suggestion",
        user_id: null,
        confidence: null,
        extend_forward: false,
      });

    render(() => (
      <PredictionSuggestion
        suggestion={() => suggestion}
        on_saved={on_saved}
        on_dismiss={on_dismiss}
      />
    ));

    fireEvent.click(screen.getByRole("button", { name: "Use suggestion" }));

    await waitFor(() => {
      expect(screen.getByRole("button", { name: "Overwrite All" })).toBeInTheDocument();
    });

    fireEvent.click(screen.getByRole("button", { name: "Overwrite All" }));

    await waitFor(() => {
      expect(vi.mocked(notification_accept)).toHaveBeenCalledTimes(2);
      expect(vi.mocked(notification_accept)).toHaveBeenLastCalledWith({
        block_start: suggestion.block_start,
        block_end: suggestion.block_end,
        label: suggestion.suggested,
        overwrite: true,
      });
    });
    expect(on_saved).toHaveBeenCalledOnce();
    expect(on_dismiss).toHaveBeenCalledOnce();
  });

  it("on overlap 409 Keep All retries with allow_overlap", async () => {
    const suggestion = suggestion_make();
    const body = {
      detail: {
        error: "overlap",
        conflicting_spans: [
          {
            start_ts: "2026-04-05T11:00:00+00:00",
            end_ts: "2026-04-05T14:00:00+00:00",
            label: "Build",
          },
        ],
      },
    };
    vi.mocked(notification_accept)
      .mockRejectedValueOnce(new Error(`409: ${JSON.stringify(body)}`))
      .mockResolvedValueOnce({
        start_ts: suggestion.block_start,
        end_ts: suggestion.block_end,
        label: suggestion.suggested,
        provenance: "suggestion",
        user_id: null,
        confidence: null,
        extend_forward: false,
      });

    render(() => <PredictionSuggestion suggestion={() => suggestion} />);

    fireEvent.click(screen.getByRole("button", { name: "Use suggestion" }));

    await waitFor(() => {
      expect(screen.getByRole("button", { name: "Keep All" })).toBeInTheDocument();
    });

    fireEvent.click(screen.getByRole("button", { name: "Keep All" }));

    await waitFor(() => {
      expect(vi.mocked(notification_accept)).toHaveBeenCalledTimes(2);
      expect(vi.mocked(notification_accept)).toHaveBeenLastCalledWith({
        block_start: suggestion.block_start,
        block_end: suggestion.block_end,
        label: suggestion.suggested,
        allow_overlap: true,
      });
    });
  });
});
