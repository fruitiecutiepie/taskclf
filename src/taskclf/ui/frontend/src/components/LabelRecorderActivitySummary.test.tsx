import { fireEvent, render, screen, waitFor } from "@solidjs/testing-library";
import { describe, expect, it, vi } from "vitest";
import {
  activity_summary_get,
  core_labels_list,
  current_label_get,
  label_create,
  label_update,
  labels_list,
} from "../lib/api";
import { LabelRecorder } from "./LabelRecorder";

vi.mock("../lib/api", () => ({
  activity_summary_get: vi.fn(),
  core_labels_list: vi.fn(),
  current_label_get: vi.fn(),
  label_create: vi.fn(),
  label_update: vi.fn(),
  labels_list: vi.fn(),
}));

describe("LabelRecorder activity summary", () => {
  it("keeps manual labeling interactive when the provider is unavailable", async () => {
    vi.mocked(core_labels_list).mockResolvedValue(["Build", "Write"]);
    vi.mocked(current_label_get).mockResolvedValue(null);
    vi.mocked(labels_list).mockResolvedValue([]);
    vi.mocked(label_update).mockResolvedValue({
      start_ts: "2026-04-09T10:00:00Z",
      end_ts: "2026-04-09T10:00:00Z",
      label: "Build",
      provenance: "manual",
      user_id: null,
      confidence: null,
      extend_forward: false,
    });
    vi.mocked(label_create).mockResolvedValue({
      start_ts: "2026-04-09T09:59:00Z",
      end_ts: "2026-04-09T10:00:00Z",
      label: "Build",
      provenance: "manual",
      user_id: null,
      confidence: 1,
      extend_forward: false,
    });
    vi.mocked(activity_summary_get).mockResolvedValue({
      activity_provider: {
        provider_id: "activitywatch",
        provider_name: "ActivityWatch",
        state: "setup_required",
        summary_available: false,
        endpoint: "http://localhost:5600",
        source_id: null,
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
      },
      recent_apps: [],
      top_apps: [],
      mean_keys_per_min: null,
      mean_clicks_per_min: null,
      mean_scroll_per_min: null,
      total_buckets: 0,
      session_count: 0,
      range_state: "provider_unavailable",
      message:
        "Manual labeling still works, but activity summaries and automatic activity tracking are unavailable until this source is set up.",
    });

    render(() => <LabelRecorder on_collapse={() => {}} />);

    fireEvent.click(await screen.findByRole("button", { name: "1m" }));

    expect(await screen.findByText("Activity source unavailable")).toBeInTheDocument();

    fireEvent.click(await screen.findByRole("button", { name: "Build" }));

    await waitFor(() => {
      expect(label_create).toHaveBeenCalled();
    });
  });
});
