import { fireEvent, render } from "@solidjs/testing-library";
import { describe, expect, it, vi } from "vitest";
import { ws_store_stub } from "../test/ws_store_stub";

describe("LabelRecorderWindow", () => {
  it("shows transition notifications for prompt events in standalone label mode", async () => {
    vi.resetModules();
    window.localStorage.clear();

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
    const notification_permission_ensure = vi.fn();
    const transition_notification_show = vi.fn();

    vi.doMock("../lib/host", () => ({
      host: {
        kind: "browser",
        isNativeWindow: false,
        invoke: vi.fn(),
      },
    }));
    vi.doMock("../lib/ws", () => ({
      ws_store_new: () => ({
        ...ws_store_stub(),
        latest_prompt: () => prompt,
      }),
    }));
    vi.doMock("../lib/notifications", () => ({
      notification_permission_ensure,
      transition_notification_show,
    }));
    vi.doMock("./LabelRecorder", () => ({
      LabelRecorder: () => <div>Label Recorder</div>,
    }));

    const { LabelRecorderWindow } = await import("./LabelRecorderWindow");
    render(() => <LabelRecorderWindow />);

    expect(notification_permission_ensure).toHaveBeenCalledOnce();
    expect(transition_notification_show).toHaveBeenCalledWith(
      prompt,
      expect.any(Function),
    );
  });

  it("forwards hover to host cancel/hide commands", async () => {
    vi.resetModules();
    const host_invoke = vi.fn().mockResolvedValue(undefined);
    vi.doMock("../lib/host", () => ({
      host: {
        kind: "electron",
        isNativeWindow: true,
        invoke: host_invoke,
      },
    }));
    vi.doMock("../lib/ws", () => ({
      ws_store_new: ws_store_stub,
    }));
    vi.doMock("./LabelRecorder", () => ({
      LabelRecorder: () => <div>Label Recorder</div>,
    }));

    const { LabelRecorderWindow } = await import("./LabelRecorderWindow");
    const { container } = render(() => <LabelRecorderWindow />);
    const root = container.firstElementChild as HTMLElement;

    fireEvent.mouseEnter(root);
    expect(host_invoke).toHaveBeenCalledWith({ cmd: "cancelLabelHide" });

    fireEvent.mouseLeave(root);
    expect(host_invoke).toHaveBeenCalledWith({ cmd: "hideLabelGrid" });
  });

  it("renders pywebview drag region on the grab strip when host is pywebview", async () => {
    vi.resetModules();
    vi.doMock("../lib/host", () => ({
      host: {
        kind: "pywebview",
        isNativeWindow: true,
        invoke: vi.fn(),
      },
    }));
    vi.doMock("../lib/ws", () => ({
      ws_store_new: ws_store_stub,
    }));
    vi.doMock("./LabelRecorder", () => ({
      LabelRecorder: () => <div>Label Recorder</div>,
    }));

    const { LabelRecorderWindow } = await import("./LabelRecorderWindow");
    const { container } = render(() => <LabelRecorderWindow />);

    expect(
      container.querySelectorAll(".pywebview-drag-region").length,
    ).toBeGreaterThanOrEqual(1);
  });
});
