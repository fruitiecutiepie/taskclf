import { render, screen } from "@solidjs/testing-library";
import { describe, expect, it, vi } from "vitest";
import { PredictionBadge } from "./PredictionBadge";

vi.mock("./ConnectionDot", () => ({
  ConnectionDot: () => <div data-testid="connection-dot" />,
}));

describe("PredictionBadge", () => {
  const base_props = {
    status: () => "connected" as const,
    latest_status: () => ({
      type: "status" as const,
      state: "idle" as const,
      current_app: "unknown",
      current_app_since: null,
      candidate_app: null,
      candidate_duration_s: 0,
      transition_threshold_s: 0,
      poll_seconds: 0,
      poll_count: 0,
      last_poll_ts: new Date(0).toISOString(),
      uptime_s: 0,
      aw_connected: false,
      aw_bucket_id: null,
      aw_host: "http://localhost:5600",
      last_event_count: 0,
      last_app_counts: {},
    }),
    latest_tray_state: () => ({
      type: "tray_state" as const,
      model_loaded: true,
      model_dir: null,
      model_schema_hash: null,
      suggested_label: null,
      suggested_confidence: null,
      transition_count: 0,
      last_transition: null,
      labels_saved_count: 0,
      data_dir: "~/.taskclf",
      ui_port: 0,
      dev_mode: false,
      paused: false,
    }),
    active_suggestion: () => null,
  };

  it("falls back to live status when there is no latest prediction", () => {
    render(() => (
      <PredictionBadge
        {...base_props}
        latest_prediction={() => null}
        live_status={() => ({
          type: "live_status",
          label: "Write",
          text: "Now: Write",
          ts: "2026-04-05T10:00:00Z",
        })}
      />
    ));

    expect(screen.getByRole("button", { name: "Write" })).toBeInTheDocument();
  });

  it("keeps explicit predictions ahead of live status", () => {
    render(() => (
      <PredictionBadge
        {...base_props}
        latest_prediction={() => ({
          type: "prediction",
          label: "Build",
          mapped_label: "Build",
          confidence: 1,
          ts: "2026-04-05T10:01:00Z",
          provenance: "manual",
        })}
        live_status={() => ({
          type: "live_status",
          label: "Write",
          text: "Now: Write",
          ts: "2026-04-05T10:00:00Z",
        })}
      />
    ));

    expect(screen.getByRole("button", { name: "Build" })).toBeInTheDocument();
  });
});
