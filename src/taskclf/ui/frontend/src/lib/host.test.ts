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

  it("falls back to browser mode without a native bridge", async () => {
    const { host } = await import("./host");

    expect(host.kind).toBe("browser");
    expect(host.isNativeWindow).toBe(false);
    await expect(host.invoke({ cmd: "toggleDashboard" })).resolves.toBeUndefined();
  });
});
