import { afterEach, describe, expect, it, vi } from "vitest";
import type { PromptLabelEvent } from "./ws";

const prompt: PromptLabelEvent = {
  type: "prompt_label",
  prev_app: "Editor",
  new_app: "Browser",
  block_start: "2026-04-05T01:00:00.000Z",
  block_end: "2026-04-05T01:05:00.000Z",
  duration_min: 5,
  suggested_label: "Research",
  suggestion_text: "Was this Research? 11:00-11:05",
};

afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

describe("transition_notification_show", () => {
  it("requests an interactive transition notification after permission is granted", async () => {
    vi.resetModules();

    const on_click = vi.fn();
    const window_focus = vi.fn();
    const notification_close = vi.fn();
    const notification_ctor = vi.fn(function NotificationMock(this: {
      close: typeof notification_close;
      onclick: (() => void) | null;
    }) {
      this.close = notification_close;
      this.onclick = null;
    });

    Object.assign(notification_ctor, {
      permission: "default",
      requestPermission: vi.fn().mockResolvedValue("granted"),
    });

    vi.stubGlobal("Notification", notification_ctor);
    vi.stubGlobal("focus", window_focus);

    const { notification_permission_ensure, transition_notification_show } =
      await import("./notifications");

    await notification_permission_ensure();
    const notification = transition_notification_show(prompt, on_click);

    expect(notification_ctor).toHaveBeenCalledWith("taskclf — Activity changed", {
      body: "Was this Research? 11:00-11:05",
      tag: "taskclf-transition",
      renotify: true,
      requireInteraction: true,
    });
    expect(notification).not.toBeNull();

    notification?.onclick?.(new MouseEvent("click"));

    expect(window_focus).toHaveBeenCalledOnce();
    expect(on_click).toHaveBeenCalledOnce();
    expect(notification_close).toHaveBeenCalledOnce();
  });

  it("returns null when notification permission is unavailable", async () => {
    vi.resetModules();

    const notification_ctor = vi.fn();
    Object.assign(notification_ctor, {
      permission: "denied",
      requestPermission: vi.fn(),
    });

    vi.stubGlobal("Notification", notification_ctor);

    const { transition_notification_show } = await import("./notifications");

    expect(transition_notification_show(prompt, vi.fn())).toBeNull();
    expect(notification_ctor).not.toHaveBeenCalled();
  });
});
