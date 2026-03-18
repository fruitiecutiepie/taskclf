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

export type HostCommand =
  | { cmd: "showLabelGrid" }
  | { cmd: "hideLabelGrid" }
  | { cmd: "toggleLabelGrid" }
  | { cmd: "cancelLabelHide" }
  | { cmd: "hideWindow" }
  | { cmd: "toggleDashboard" }
  | { cmd: "toggleStatePanel" }
  | { cmd: "showStatePanel" }
  | { cmd: "hideStatePanel" }
  | { cmd: "cancelPanelHide" }
  | { cmd: "frontendDebugLog"; message: string }
  | { cmd: "frontendErrorLog"; message: string };

export type Host = {
  invoke(command: HostCommand): Promise<void>;
  readonly isNativeWindow: boolean;
};

declare global {
  interface Window {
    pywebview?: {
      api?: {
        label_grid_show(): Promise<void>;
        label_grid_hide(): Promise<void>;
        label_grid_toggle(): Promise<void>;
        label_grid_cancel_hide(): Promise<void>;
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
  get isNativeWindow(): boolean {
    return pywebview_api_ref() !== null;
  }

  async invoke(command: HostCommand): Promise<void> {
    const api = pywebview_api_ref();
    if (!api) {
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
        case "hideWindow":
          await api.window_hide();
          break;
        case "toggleDashboard":
          await api.dashboard_toggle();
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
      // Show/hide may fail transiently; don't block the caller.
    }
  }
}

export const host: Host = new AdaptiveHost();
