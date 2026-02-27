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

class PyWebViewHost implements Host {
  readonly isNativeWindow = true;

  async invoke(command: HostCommand): Promise<void> {
    const api = window.pywebview?.api;
    if (!api) return;
    switch (command.cmd) {
      case "setCompact":
        return api.set_compact();
      case "setExpanded":
        return api.set_expanded();
      case "hideWindow":
        return api.hide_window();
    }
  }
}

class BrowserHost implements Host {
  readonly isNativeWindow = false;

  async invoke(_command: HostCommand): Promise<void> {
    // No-op in browser mode -- window control is not available.
  }
}

function detectHost(): Host {
  if (typeof window !== "undefined" && window.pywebview) {
    return new PyWebViewHost();
  }
  return new BrowserHost();
}

export const host: Host = detectHost();
