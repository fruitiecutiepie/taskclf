import { render, screen } from "@solidjs/testing-library";
import { describe, expect, it } from "vitest";
import { StatusActivityWatch } from "./StatusActivityWatch";

describe("StatusActivityWatch", () => {
  it("renders provider-neutral activity source setup guidance", () => {
    render(() => (
      <StatusActivityWatch
        status={() => ({
          type: "status",
          state: "idle",
          current_app: "unknown",
          current_app_since: null,
          candidate_app: null,
          candidate_duration_s: 0,
          transition_threshold_s: 0,
          poll_seconds: 0,
          poll_count: 0,
          last_poll_ts: new Date(0).toISOString(),
          uptime_s: 0,
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
          aw_connected: false,
          aw_bucket_id: null,
          aw_host: "http://localhost:5600",
          last_event_count: 0,
          last_app_counts: {},
        })}
      />
    ));

    expect(screen.getByText("Activity Source")).toBeInTheDocument();
    expect(screen.getByText("Activity source unavailable")).toBeInTheDocument();
    expect(
      screen.getByText(/Manual labeling still works, but activity summaries/),
    ).toBeInTheDocument();
  });
});
