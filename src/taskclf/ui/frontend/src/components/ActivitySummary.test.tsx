import { render, screen, waitFor } from "@solidjs/testing-library";
import { describe, expect, it, vi } from "vitest";
import type { ActivityProviderStatus, ActivitySummary as ActivitySummaryData } from "../lib/api";
import { activity_summary_get } from "../lib/api";
import { ActivitySummary } from "./ActivitySummary";

vi.mock("../lib/api", () => ({
  activity_summary_get: vi.fn(),
}));

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

describe("ActivitySummary", () => {
  const range = () => ({
    start: "2026-04-09T10:00:00Z",
    end: "2026-04-09T10:05:00Z",
  });

  it("shows a no-data message for empty ranges", async () => {
    vi.mocked(activity_summary_get).mockResolvedValueOnce(activity_summary_make());

    render(() => <ActivitySummary time_range={range} show_empty />);

    expect(await screen.findByText("No activity data for this window")).toBeInTheDocument();
  });

  it("shows setup guidance when the provider is unavailable", async () => {
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

    render(() => <ActivitySummary time_range={range} show_empty />);

    expect(await screen.findByText("Activity source unavailable")).toBeInTheDocument();
    expect(
      screen.getByText(/Manual labeling still works, but activity summaries/),
    ).toBeInTheDocument();
  });

  it("shows a generic fallback when the summary request fails", async () => {
    vi.mocked(activity_summary_get).mockRejectedValueOnce(new Error("boom"));

    render(() => <ActivitySummary time_range={range} show_empty />);

    await waitFor(() => {
      expect(
        screen.getByText("Activity summary unavailable right now"),
      ).toBeInTheDocument();
    });
  });
});
