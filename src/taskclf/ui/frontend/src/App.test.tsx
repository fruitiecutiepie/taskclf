import { fireEvent, render, screen } from "@solidjs/testing-library";
import { describe, expect, it, vi } from "vitest";
import { ws_store_stub } from "./test/ws_store_stub";

describe("App view routing", () => {
  it("renders compact shell when view query is absent", async () => {
    vi.resetModules();
    window.history.replaceState({}, "", "/");

    vi.doMock("./lib/host", () => ({
      host: {
        kind: "browser",
        isNativeWindow: false,
        invoke: vi.fn(),
      },
    }));
    vi.doMock("./lib/log", () => ({
      frontend_error_handlers_install: () => () => {},
    }));
    vi.doMock("./lib/notifications", () => ({
      notification_permission_ensure: vi.fn(),
      transition_notification_show: vi.fn(),
    }));
    vi.doMock("./lib/ws", () => ({
      ws_store_new: ws_store_stub,
    }));

    const { default: App } = await import("./App");
    render(() => <App />);

    expect(screen.getByRole("button", { name: "No Model" })).toBeInTheDocument();
  });

  it("shows browser label popup on hover with deferred hide", async () => {
    vi.resetModules();
    vi.useFakeTimers();
    window.history.replaceState({}, "", "/");

    vi.doMock("./lib/host", () => ({
      host: {
        kind: "browser",
        isNativeWindow: false,
        invoke: vi.fn(),
      },
    }));
    vi.doMock("./lib/log", () => ({
      frontend_error_handlers_install: () => () => {},
    }));
    vi.doMock("./lib/notifications", () => ({
      notification_permission_ensure: vi.fn(),
      transition_notification_show: vi.fn(),
    }));
    vi.doMock("./lib/ws", () => ({
      ws_store_new: ws_store_stub,
    }));
    vi.doMock("./components/LabelRecorder", () => ({
      LabelRecorder: () => <div data-testid="inline-label-grid">Label Recorder</div>,
    }));
    vi.doMock("./components/StatusPanel", () => ({
      StatusPanel: () => <div>Status Panel</div>,
    }));

    const { default: App } = await import("./App");
    render(() => <App />);

    const badge_button = screen.getByRole("button", { name: "No Model" });
    fireEvent.mouseEnter(badge_button);
    expect(screen.getByTestId("inline-label-grid")).toBeInTheDocument();

    fireEvent.mouseLeave(badge_button);
    expect(screen.getByTestId("inline-label-grid")).toBeInTheDocument();

    vi.advanceTimersByTime(299);
    expect(screen.getByTestId("inline-label-grid")).toBeInTheDocument();

    vi.advanceTimersByTime(1);
    expect(screen.queryByTestId("inline-label-grid")).not.toBeInTheDocument();
    vi.useRealTimers();
  });

  it("renders label view when view=label", async () => {
    vi.resetModules();
    window.history.replaceState({}, "", "/?view=label");

    vi.doMock("./lib/host", () => ({
      host: {
        kind: "browser",
        isNativeWindow: false,
        invoke: vi.fn(),
      },
    }));
    vi.doMock("./lib/log", () => ({
      frontend_error_handlers_install: () => () => {},
    }));
    vi.doMock("./lib/notifications", () => ({
      notification_permission_ensure: vi.fn(),
      transition_notification_show: vi.fn(),
    }));
    vi.doMock("./lib/ws", () => ({
      ws_store_new: ws_store_stub,
    }));
    vi.doMock("./components/LabelRecorder", () => ({
      LabelRecorder: () => <div data-testid="label-recorder-route">Label Recorder</div>,
    }));

    const { default: App } = await import("./App");
    render(() => <App />);

    expect(screen.getByTestId("label-recorder-route")).toBeInTheDocument();
  });

  it("renders panel view when view=panel", async () => {
    vi.resetModules();
    window.history.replaceState({}, "", "/?view=panel");

    vi.doMock("./lib/host", () => ({
      host: {
        kind: "browser",
        isNativeWindow: false,
        invoke: vi.fn(),
      },
    }));
    vi.doMock("./lib/log", () => ({
      frontend_error_handlers_install: () => () => {},
    }));
    vi.doMock("./lib/notifications", () => ({
      notification_permission_ensure: vi.fn(),
      transition_notification_show: vi.fn(),
    }));
    vi.doMock("./lib/ws", () => ({
      ws_store_new: ws_store_stub,
    }));
    vi.doMock("./components/StatusPanel", () => ({
      StatusPanel: () => <div data-testid="status-panel-route">Status Panel</div>,
    }));

    const { default: App } = await import("./App");
    render(() => <App />);

    expect(screen.getByTestId("status-panel-route")).toBeInTheDocument();
  });
});

describe("App drag regions", () => {
  it("keeps the label badge outside every pywebview drag region", async () => {
    vi.resetModules();
    window.history.replaceState({}, "", "/");

    const host_invoke = vi.fn();

    vi.doMock("./lib/host", () => ({
      host: {
        kind: "pywebview",
        isNativeWindow: true,
        invoke: host_invoke,
      },
    }));
    vi.doMock("./lib/log", () => ({
      frontend_error_handlers_install: () => () => {},
    }));
    vi.doMock("./lib/notifications", () => ({
      notification_permission_ensure: vi.fn(),
      transition_notification_show: vi.fn(),
    }));
    vi.doMock("./lib/ws", () => ({
      ws_store_new: () => ({
        latest_status: () => ({
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
            state: "checking",
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
        }),
        latest_prediction: () => null,
        latest_tray_state: () => ({
          type: "tray_state",
          model_loaded: false,
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
        latest_prompt: () => null,
        live_status: () => null,
        label_grid_requested: () => 0,
        connection_status: () => "connected",
        ws_stats: () => ({
          message_count: 0,
          status_count: 0,
          prediction_count: 0,
          tray_state_count: 0,
          suggestion_count: 0,
          last_message_at: null,
          reconnect_count: 0,
          connected_since: null,
        }),
        train_state: () => ({
          job_id: null,
          status: "idle",
          step: null,
          progress_pct: null,
          message: null,
          error: null,
          metrics: null,
          model_dir: null,
        }),
        suggestion_dismiss: vi.fn(),
      }),
    }));
    vi.doMock("./components/LabelRecorder", () => ({
      LabelRecorder: () => <div>Label Recorder</div>,
    }));
    vi.doMock("./components/StatusPanel", () => ({
      StatusPanel: () => <div>Status Panel</div>,
    }));

    const { default: App } = await import("./App");
    render(() => <App />);

    const drag_regions = document.querySelectorAll(".pywebview-drag-region");
    expect(drag_regions.length).toBe(2);

    const badge_button = screen.getByRole("button", { name: "No Model" });
    for (const el of drag_regions) {
      expect(el).not.toContainElement(badge_button);
    }
  });

  it("uses popup host commands in Electron multi-window mode", async () => {
    vi.resetModules();
    window.history.replaceState({}, "", "/");

    const host_invoke = vi.fn().mockResolvedValue(undefined);

    vi.doMock("./lib/host", () => ({
      host: {
        kind: "electron",
        isNativeWindow: true,
        invoke: host_invoke,
      },
    }));
    vi.doMock("./lib/log", () => ({
      frontend_error_handlers_install: () => () => {},
    }));
    vi.doMock("./lib/notifications", () => ({
      notification_permission_ensure: vi.fn(),
      transition_notification_show: vi.fn(),
    }));
    vi.doMock("./lib/ws", () => ({
      ws_store_new: () => ({
        latest_status: () => ({
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
            state: "checking",
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
        }),
        latest_prediction: () => null,
        latest_tray_state: () => ({
          type: "tray_state",
          model_loaded: false,
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
        latest_prompt: () => null,
        live_status: () => null,
        label_grid_requested: () => 0,
        connection_status: () => "connected",
        ws_stats: () => ({
          message_count: 0,
          status_count: 0,
          prediction_count: 0,
          tray_state_count: 0,
          suggestion_count: 0,
          last_message_at: null,
          reconnect_count: 0,
          connected_since: null,
        }),
        train_state: () => ({
          job_id: null,
          status: "idle",
          step: null,
          progress_pct: null,
          message: null,
          error: null,
          metrics: null,
          model_dir: null,
        }),
        suggestion_dismiss: vi.fn(),
      }),
    }));

    const { default: App } = await import("./App");
    render(() => <App />);

    expect(host_invoke).not.toHaveBeenCalledWith(
      expect.objectContaining({ cmd: "setWindowMode" }),
    );

    const badge_button = screen.getByRole("button", { name: "No Model" });
    fireEvent.mouseEnter(badge_button);

    expect(host_invoke).toHaveBeenCalledWith({ cmd: "showLabelGrid" });

    fireEvent.click(badge_button);

    expect(host_invoke).toHaveBeenCalledWith({ cmd: "toggleLabelGrid" });
  });

  it("routes transition prompts through Electron native notifications", async () => {
    vi.resetModules();
    window.history.replaceState({}, "", "/");

    const host_invoke = vi.fn().mockResolvedValue(undefined);
    const notification_permission_ensure = vi.fn();
    const transition_notification_show = vi.fn();
    const prompt = {
      type: "prompt_label" as const,
      prev_app: "Editor",
      new_app: "Browser",
      block_start: "2026-04-05T00:00:00.000Z",
      block_end: "2026-04-05T00:05:00.000Z",
      duration_min: 5,
      suggested_label: "ReadResearch",
      suggestion_text: "Was this ReadResearch? 10:00–10:05",
    };

    vi.doMock("./lib/host", () => ({
      host: {
        kind: "electron",
        isNativeWindow: true,
        invoke: host_invoke,
      },
    }));
    vi.doMock("./lib/log", () => ({
      frontend_error_handlers_install: () => () => {},
    }));
    vi.doMock("./lib/notifications", () => ({
      notification_permission_ensure,
      transition_notification_show,
    }));
    vi.doMock("./lib/ws", () => ({
      ws_store_new: () => ({
        latest_status: () => ({
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
            state: "checking",
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
        }),
        latest_prediction: () => null,
        latest_tray_state: () => ({
          type: "tray_state",
          model_loaded: false,
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
        latest_prompt: () => prompt,
        live_status: () => null,
        label_grid_requested: () => 0,
        connection_status: () => "connected",
        ws_stats: () => ({
          message_count: 0,
          status_count: 0,
          prediction_count: 0,
          tray_state_count: 0,
          suggestion_count: 0,
          last_message_at: null,
          reconnect_count: 0,
          connected_since: null,
        }),
        train_state: () => ({
          job_id: null,
          status: "idle",
          step: null,
          progress_pct: null,
          message: null,
          error: null,
          metrics: null,
          model_dir: null,
        }),
        suggestion_dismiss: vi.fn(),
      }),
    }));

    const { default: App } = await import("./App");
    render(() => <App />);

    expect(notification_permission_ensure).not.toHaveBeenCalled();
    expect(transition_notification_show).not.toHaveBeenCalled();
    expect(host_invoke).toHaveBeenCalledWith({
      cmd: "showTransitionNotification",
      prompt,
    });
  });
});
