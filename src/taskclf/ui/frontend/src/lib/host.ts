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
  | { cmd: "setCompact" }
  | { cmd: "setExpanded" }
  | { cmd: "hideWindow" };

export interface Host {
  invoke(command: HostCommand): Promise<void>;
  readonly isNativeWindow: boolean;
}

declare global {
  interface Window {
    pywebview?: {
      api?: {
        set_compact(): Promise<void>;
        set_expanded(): Promise<void>;
        hide_window(): Promise<void>;
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
        case "setCompact":
          await api.set_compact();
          break;
        case "setExpanded":
          await api.set_expanded();
          break;
        case "hideWindow":
          await api.hide_window();
          break;
      }
    } catch {
      // Resize/hide may fail transiently; don't block the caller.
    }
  }
}

export const host: Host = new AdaptiveHost();
