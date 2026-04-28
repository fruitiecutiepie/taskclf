/**
 * Host API abstraction -- the migration seam between pywebview and Electron.
 *
 * All window-control calls go through this interface. Data operations
 * (labels, queue, features) stay in api.ts/ws.ts unchanged -- those are
 * pure HTTP/WebSocket and work identically in any host.
 *
 * To migrate to Electron: add an ElectronHost that maps commands to
 * ipcRenderer.invoke(). Zero component code changes needed.
 */

import type { PromptLabelEvent } from "./ws";

// #region agent log
function agent_debug_log(
  runId: string,
  hypothesisId: string,
  location: string,
  message: string,
  data: Record<string, unknown>,
) {
  fetch("http://localhost:7434/ingest/307992f9-e352-421f-9c8b-95a59cddc80f", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-Debug-Session-Id": "f37ed4",
    },
    body: JSON.stringify({
      sessionId: "f37ed4",
      runId,
      hypothesisId,
      location,
      message,
      data,
      timestamp: Date.now(),
    }),
  }).catch(() => {});
}
// #endregion

export type WindowMode = "compact" | "label" | "panel" | "dashboard";

export type HostKind = "browser" | "pywebview" | "electron";

export type HostCommand =
  | { cmd: "showLabelGrid" }
  | { cmd: "hideLabelGrid" }
  | { cmd: "toggleLabelGrid" }
  | { cmd: "cancelLabelHide" }
  | { cmd: "showTransitionNotification"; prompt: PromptLabelEvent }
  | { cmd: "hideWindow" }
  | { cmd: "toggleDashboard" }
  | { cmd: "setWindowMode"; mode: WindowMode }
  | { cmd: "toggleStatePanel" }
  | { cmd: "showStatePanel" }
  | { cmd: "hideStatePanel" }
  | { cmd: "cancelPanelHide" }
  | { cmd: "frontendDebugLog"; message: string }
  | { cmd: "frontendErrorLog"; message: string };

export type Host = {
  invoke(command: HostCommand): Promise<void>;
  readonly isNativeWindow: boolean;
  readonly kind: HostKind;
};

declare global {
  interface Window {
    electronHost?: {
      invoke(command: HostCommand): Promise<void>;
    };
    pywebview?: {
      api?: {
        label_grid_show(): Promise<void>;
        label_grid_hide(): Promise<void>;
        label_grid_toggle(): Promise<void>;
        label_grid_cancel_hide(): Promise<void>;
        show_transition_notification(prompt: PromptLabelEvent): Promise<void>;
        window_hide(): Promise<void>;
        dashboard_toggle(): Promise<void>;
        state_panel_toggle(): Promise<void>;
        state_panel_show(): Promise<void>;
        state_panel_hide(): Promise<void>;
        state_panel_cancel_hide(): Promise<void>;
        frontend_debug_log(message: string): Promise<void>;
        frontend_error_log(message: string): Promise<void>;
      };
    };
  }
}

function electron_api_ref() {
  return window.electronHost ?? null;
}

function pywebview_api_ref() {
  return window.pywebview?.api ?? null;
}

/**
 * Lazy-detecting host that re-checks for pywebview on every invoke.
 *
 * pywebview injects `window.pywebview` after the page's initial JS runs,
 * so a one-shot check at module load time would miss it.
 */
class AdaptiveHost implements Host {
  get kind(): HostKind {
    if (electron_api_ref() !== null) {
      return "electron";
    }
    if (pywebview_api_ref() !== null) {
      return "pywebview";
    }
    return "browser";
  }

  get isNativeWindow(): boolean {
    return this.kind !== "browser";
  }

  async invoke(command: HostCommand): Promise<void> {
    const electron_api = electron_api_ref();
    if (command.cmd === "showTransitionNotification") {
      // #region agent log
      agent_debug_log(
        "pre-fix",
        "H4",
        "src/taskclf/ui/frontend/src/lib/host.ts:94",
        "host invoke received transition notification command",
        {
          host_kind: this.kind,
          has_electron_api: electron_api !== null,
          has_pywebview_api: pywebview_api_ref() !== null,
          href: window.location.href,
          block_start: command.prompt.block_start,
          block_end: command.prompt.block_end,
          suggested_label: command.prompt.suggested_label,
        },
      );
      // #endregion
    }
    if (electron_api) {
      try {
        await electron_api.invoke(command);
        if (command.cmd === "showTransitionNotification") {
          // #region agent log
          agent_debug_log(
            "pre-fix",
            "H4",
            "src/taskclf/ui/frontend/src/lib/host.ts:98",
            "electron transition notification invoke resolved",
            {
              href: window.location.href,
            },
          );
          // #endregion
        }
      } catch {
        if (command.cmd === "showTransitionNotification") {
          // #region agent log
          agent_debug_log(
            "pre-fix",
            "H4",
            "src/taskclf/ui/frontend/src/lib/host.ts:99",
            "electron transition notification invoke threw",
            {
              href: window.location.href,
            },
          );
          // #endregion
        }
        // Electron IPC can fail transiently while the shell is loading.
      }
      return;
    }

    const api = pywebview_api_ref();
    if (!api) {
      if (command.cmd === "showTransitionNotification") {
        // #region agent log
        agent_debug_log(
          "pre-fix",
          "H4",
          "src/taskclf/ui/frontend/src/lib/host.ts:106",
          "transition notification command dropped because native api is unavailable",
          {
            href: window.location.href,
            host_kind: this.kind,
          },
        );
        // #endregion
      }
      return;
    }
    try {
      switch (command.cmd) {
        case "showLabelGrid":
          await api.label_grid_show();
          break;
        case "hideLabelGrid":
          await api.label_grid_hide();
          break;
        case "toggleLabelGrid":
          await api.label_grid_toggle();
          break;
        case "cancelLabelHide":
          await api.label_grid_cancel_hide();
          break;
        case "showTransitionNotification":
          await api.show_transition_notification(command.prompt);
          // #region agent log
          agent_debug_log(
            "pre-fix",
            "H4",
            "src/taskclf/ui/frontend/src/lib/host.ts:124",
            "pywebview transition notification invoke resolved",
            {
              href: window.location.href,
            },
          );
          // #endregion
          break;
        case "hideWindow":
          await api.window_hide();
          break;
        case "toggleDashboard":
          await api.dashboard_toggle();
          break;
        case "setWindowMode":
          break;
        case "toggleStatePanel":
          await api.state_panel_toggle();
          break;
        case "showStatePanel":
          await api.state_panel_show();
          break;
        case "hideStatePanel":
          await api.state_panel_hide();
          break;
        case "cancelPanelHide":
          await api.state_panel_cancel_hide();
          break;
        case "frontendDebugLog":
          await api.frontend_debug_log(command.message);
          break;
        case "frontendErrorLog":
          await api.frontend_error_log(command.message);
          break;
      }
    } catch {
      if (command.cmd === "showTransitionNotification") {
        // #region agent log
        agent_debug_log(
          "pre-fix",
          "H4",
          "src/taskclf/ui/frontend/src/lib/host.ts:153",
          "pywebview transition notification invoke threw",
          {
            href: window.location.href,
          },
        );
        // #endregion
      }
      // Show/hide may fail transiently; don't block the caller.
    }
  }
}

export const host: Host = new AdaptiveHost();
