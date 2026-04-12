import { afterEach, describe, expect, it, vi } from "vitest";

function host_globals_clear() {
  delete (window as typeof window & { electronHost?: unknown }).electronHost;
  delete (window as typeof window & { pywebview?: unknown }).pywebview;
}

describe("host", () => {
  afterEach(() => {
    host_globals_clear();
    vi.resetModules();
    vi.restoreAllMocks();
  });

  it("uses the Electron bridge when available", async () => {
    const electron_invoke = vi.fn().mockResolvedValue(undefined);
    (
      window as typeof window & { electronHost?: { invoke: typeof electron_invoke } }
    ).electronHost = {
      invoke: electron_invoke,
    };

    const { host } = await import("./host");

    expect(host.kind).toBe("electron");
    expect(host.isNativeWindow).toBe(true);

    await host.invoke({ cmd: "setWindowMode", mode: "panel" });

    expect(electron_invoke).toHaveBeenCalledOnce();
    expect(electron_invoke).toHaveBeenCalledWith({
      cmd: "setWindowMode",
      mode: "panel",
    });
  });

  it("uses the pywebview bridge when available", async () => {
    const dashboard_toggle = vi.fn().mockResolvedValue(undefined);
    (
      window as typeof window & {
        pywebview?: { api?: { dashboard_toggle: typeof dashboard_toggle } };
      }
    ).pywebview = {
      api: {
        dashboard_toggle,
      } as never,
    };

    const { host } = await import("./host");

    expect(host.kind).toBe("pywebview");
    expect(host.isNativeWindow).toBe(true);

    await host.invoke({ cmd: "toggleDashboard" });

    expect(dashboard_toggle).toHaveBeenCalledOnce();
  });

  it("forwards transition notifications through the pywebview bridge", async () => {
    const show_transition_notification = vi.fn().mockResolvedValue(undefined);
    const prompt = {
      type: "prompt_label" as const,
      prev_app: "Editor",
      new_app: "Browser",
      block_start: "2026-04-05T00:00:00.000Z",
      block_end: "2026-04-05T00:05:00.000Z",
      duration_min: 5,
      suggested_label: "ReadResearch",
      suggestion_text: "Was this ReadResearch? 10:00-10:05",
    };
    (
      window as typeof window & {
        pywebview?: {
          api?: {
            show_transition_notification: typeof show_transition_notification;
          };
        };
      }
    ).pywebview = {
      api: {
        show_transition_notification,
      } as never,
    };

    const { host } = await import("./host");

    await host.invoke({ cmd: "showTransitionNotification", prompt });

    expect(show_transition_notification).toHaveBeenCalledOnce();
    expect(show_transition_notification).toHaveBeenCalledWith(prompt);
  });

  it("falls back to browser mode without a native bridge", async () => {
    const { host } = await import("./host");

    expect(host.kind).toBe("browser");
    expect(host.isNativeWindow).toBe(false);
    await expect(host.invoke({ cmd: "toggleDashboard" })).resolves.toBeUndefined();
  });
});
