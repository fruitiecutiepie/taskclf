import { fireEvent, render, screen, waitFor } from "@solidjs/testing-library";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { ErrorBanner } from "./ErrorBanner";

const clipboard_write = vi.fn().mockResolvedValue(undefined);

describe("ErrorBanner", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    Object.defineProperty(window.navigator, "clipboard", {
      configurable: true,
      value: { writeText: clipboard_write },
    });
  });

  it("copies the current error text", async () => {
    render(() => <ErrorBanner message="broken" />);

    fireEvent.click(screen.getByRole("button", { name: "Copy error" }));

    await waitFor(() => expect(clipboard_write).toHaveBeenCalledWith("broken"));
    await waitFor(() =>
      expect(screen.getByRole("button", { name: "Copied" })).toBeInTheDocument(),
    );
  });

  it("calls the close handler when dismissed", () => {
    const on_close = vi.fn();

    render(() => <ErrorBanner message="broken" on_close={on_close} />);

    fireEvent.click(screen.getByRole("button", { name: "Close error" }));

    expect(on_close).toHaveBeenCalledTimes(1);
  });
});
