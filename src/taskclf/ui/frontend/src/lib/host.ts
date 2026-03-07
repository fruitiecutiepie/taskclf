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
  | { cmd: "cancelPanelHide" };

export interface Host {
  invoke(command: HostCommand): Promise<void>;
  readonly isNativeWindow: boolean;
}

declare global {
  interface Window {
    pywebview?: {
      api?: {
        show_label_grid(): Promise<void>;
        hide_label_grid(): Promise<void>;
        toggle_label_grid(): Promise<void>;
        cancel_label_hide(): Promise<void>;
        hide_window(): Promise<void>;
        toggle_dashboard(): Promise<void>;
        toggle_state_panel(): Promise<void>;
        show_state_panel(): Promise<void>;
        hide_state_panel(): Promise<void>;
        cancel_panel_hide(): Promise<void>;
      };
    };
  }
}

function getPyWebViewApi() {
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
    return getPyWebViewApi() !== null;
  }

  async invoke(command: HostCommand): Promise<void> {
    const api = getPyWebViewApi();
    if (!api) return;
    try {
      switch (command.cmd) {
        case "showLabelGrid":
          await api.show_label_grid();
          break;
        case "hideLabelGrid":
          await api.hide_label_grid();
          break;
        case "toggleLabelGrid":
          await api.toggle_label_grid();
          break;
        case "cancelLabelHide":
          await api.cancel_label_hide();
          break;
        case "hideWindow":
          await api.hide_window();
          break;
        case "toggleDashboard":
          await api.toggle_dashboard();
          break;
        case "toggleStatePanel":
          await api.toggle_state_panel();
          break;
        case "showStatePanel":
          await api.show_state_panel();
          break;
        case "hideStatePanel":
          await api.hide_state_panel();
          break;
        case "cancelPanelHide":
          await api.cancel_panel_hide();
          break;
      }
    } catch {
      // Show/hide may fail transiently; don't block the caller.
    }
  }
}

export const host: Host = new AdaptiveHost();
