import { fireEvent, render, screen } from "@solidjs/testing-library";
import { describe, expect, it, vi } from "vitest";

describe("App native pill", () => {
  it("keeps the label badge outside the pywebview drag region", async () => {
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

    const drag_region = document.querySelector(".pywebview-drag-region");
    expect(drag_region).not.toBeNull();

    const badge_button = screen.getByRole("button", { name: "No Model" });
    expect(drag_region).not.toContainElement(badge_button);
  });

  it("syncs Electron window mode as single-window panels open", async () => {
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

    expect(host_invoke).toHaveBeenCalledWith({
      cmd: "setWindowMode",
      mode: "compact",
    });

    const badge_button = screen.getByRole("button", { name: "No Model" });
    fireEvent.mouseEnter(badge_button);

    expect(host_invoke).toHaveBeenCalledWith({
      cmd: "setWindowMode",
      mode: "label",
    });
  });
});
