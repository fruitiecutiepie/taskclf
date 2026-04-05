import { fireEvent, render, screen } from "@solidjs/testing-library";
import { describe, expect, it, vi } from "vitest";
import { ws_store_stub } from "../test/ws_store_stub";

describe("StatusPanelWindow", () => {
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
    vi.doMock("./StatusPanel", () => ({
      StatusPanel: () => <div>Status Panel</div>,
    }));

    const { StatusPanelWindow } = await import("./StatusPanelWindow");
    const { container } = render(() => <StatusPanelWindow />);
    const root = container.firstElementChild as HTMLElement;

    fireEvent.mouseEnter(root);
    expect(host_invoke).toHaveBeenCalledWith({ cmd: "cancelPanelHide" });

    fireEvent.mouseLeave(root);
    expect(host_invoke).toHaveBeenCalledWith({ cmd: "hideStatePanel" });
  });

  it("invokes showLabelGrid from on_open_label_recorder", async () => {
    vi.resetModules();
    const host_invoke = vi.fn().mockResolvedValue(undefined);
    vi.doMock("../lib/host", () => ({
      host: {
        kind: "browser",
        isNativeWindow: false,
        invoke: host_invoke,
      },
    }));
    vi.doMock("../lib/ws", () => ({
      ws_store_new: ws_store_stub,
    }));
    vi.doMock("./StatusPanel", () => ({
      StatusPanel: (props: { on_open_label_recorder: () => void }) => (
        <button type="button" onClick={() => props.on_open_label_recorder()}>
          Open label recorder
        </button>
      ),
    }));

    const { StatusPanelWindow } = await import("./StatusPanelWindow");
    render(() => <StatusPanelWindow />);

    fireEvent.click(screen.getByRole("button", { name: "Open label recorder" }));
    expect(host_invoke).toHaveBeenCalledWith({ cmd: "showLabelGrid" });
  });
});
