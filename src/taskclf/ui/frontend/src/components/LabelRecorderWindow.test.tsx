import { fireEvent, render } from "@solidjs/testing-library";
import { describe, expect, it, vi } from "vitest";
import { ws_store_stub } from "../test/ws_store_stub";

describe("LabelRecorderWindow", () => {
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
