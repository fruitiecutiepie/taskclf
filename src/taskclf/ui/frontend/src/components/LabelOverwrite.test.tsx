import { fireEvent, render, screen } from "@solidjs/testing-library";
import { describe, expect, it, vi } from "vitest";
import { LabelOverwrite, type OverwritePending } from "./LabelOverwrite";

function pending_make(overrides: Partial<OverwritePending> = {}): OverwritePending {
  return {
    label: "Write",
    start: "2026-03-14T10:00:00+00:00",
    end: "2026-03-14T11:00:00+00:00",
    conflicts: [
      {
        start_ts: "2026-03-14T09:30:00+00:00",
        end_ts: "2026-03-14T10:30:00+00:00",
        label: "Review",
      },
    ],
    confidence: 1,
    extend_forward: false,
    ...overrides,
  };
}

const noop = () => {};

describe("LabelOverwrite", () => {
  // ------------------------------------------------------------------
  // Sorting
  // ------------------------------------------------------------------
  it("renders conflicts sorted by start_ts", () => {
    const pending = pending_make({
      conflicts: [
        {
          start_ts: "2026-03-14T11:00:00+00:00",
          end_ts: "2026-03-14T12:00:00+00:00",
          label: "Debug",
        },
        {
          start_ts: "2026-03-14T09:00:00+00:00",
          end_ts: "2026-03-14T10:00:00+00:00",
          label: "Review",
        },
        {
          start_ts: "2026-03-14T10:00:00+00:00",
          end_ts: "2026-03-14T11:00:00+00:00",
          label: "Build",
        },
      ],
    });

    render(() => (
      <LabelOverwrite
        pending={pending}
        on_confirm={noop}
        on_keep_all={noop}
        on_cancel={noop}
      />
    ));

    fireEvent.click(screen.getByText("show details"));

    const all_labels = screen.getAllByText(/Review|Build|Debug/);
    // First 3 are the summary (sorted by first appearance), last 3 are detail rows (sorted by start_ts)
    const detail_labels = all_labels.slice(3);
    const detail_order = detail_labels.map((el) => el.textContent);
    expect(detail_order).toEqual(["Review", "Build", "Debug"]);
  });

  // ------------------------------------------------------------------
  // Single conflict: shows detail inline, no expand toggle
  // ------------------------------------------------------------------
  it("shows span detail inline for a single conflict", () => {
    const pending = pending_make();

    render(() => (
      <LabelOverwrite
        pending={pending}
        on_confirm={noop}
        on_keep_all={noop}
        on_cancel={noop}
      />
    ));

    const reviews = screen.getAllByText("Review");
    expect(reviews).toHaveLength(2); // summary label + detail row
    expect(screen.queryByText("show details")).not.toBeInTheDocument();
  });

  // ------------------------------------------------------------------
  // Multiple conflicts: compact summary with expand toggle
  // ------------------------------------------------------------------
  it("shows compact summary with expand toggle for multiple conflicts", () => {
    const pending = pending_make({
      conflicts: [
        {
          start_ts: "2026-03-14T09:00:00+00:00",
          end_ts: "2026-03-14T10:30:00+00:00",
          label: "Review",
        },
        {
          start_ts: "2026-03-14T10:00:00+00:00",
          end_ts: "2026-03-14T11:00:00+00:00",
          label: "Debug",
        },
      ],
    });

    render(() => (
      <LabelOverwrite
        pending={pending}
        on_confirm={noop}
        on_keep_all={noop}
        on_cancel={noop}
      />
    ));

    expect(screen.getByText(/2 labels:/)).toBeInTheDocument();
    expect(screen.getByText("show details")).toBeInTheDocument();
  });

  it("toggles detail visibility on click", () => {
    const pending = pending_make({
      conflicts: [
        {
          start_ts: "2026-03-14T09:00:00+00:00",
          end_ts: "2026-03-14T10:30:00+00:00",
          label: "Review",
        },
        {
          start_ts: "2026-03-14T10:00:00+00:00",
          end_ts: "2026-03-14T11:00:00+00:00",
          label: "Debug",
        },
      ],
    });

    render(() => (
      <LabelOverwrite
        pending={pending}
        on_confirm={noop}
        on_keep_all={noop}
        on_cancel={noop}
      />
    ));

    const toggle = screen.getByText("show details");
    fireEvent.click(toggle);
    expect(screen.getByText("hide details")).toBeInTheDocument();

    fireEvent.click(screen.getByText("hide details"));
    expect(screen.getByText("show details")).toBeInTheDocument();
  });

  // ------------------------------------------------------------------
  // Deduplicates label names in summary
  // ------------------------------------------------------------------
  it("deduplicates label names in the summary line", () => {
    const pending = pending_make({
      conflicts: [
        {
          start_ts: "2026-03-14T09:00:00+00:00",
          end_ts: "2026-03-14T09:30:00+00:00",
          label: "Review",
        },
        {
          start_ts: "2026-03-14T09:30:00+00:00",
          end_ts: "2026-03-14T10:30:00+00:00",
          label: "Review",
        },
        {
          start_ts: "2026-03-14T10:00:00+00:00",
          end_ts: "2026-03-14T11:00:00+00:00",
          label: "Debug",
        },
      ],
    });

    render(() => (
      <LabelOverwrite
        pending={pending}
        on_confirm={noop}
        on_keep_all={noop}
        on_cancel={noop}
      />
    ));

    const summary_row = screen.getByText(/3 labels:/);
    expect(summary_row.parentElement?.textContent).toContain("Review");
    expect(summary_row.parentElement?.textContent).toContain("Debug");
    const summary_text = summary_row.parentElement?.textContent ?? "";
    const review_count = summary_text.split("Review").length - 1;
    expect(review_count).toBe(1);
  });

  // ------------------------------------------------------------------
  // Shows new label name in affected range
  // ------------------------------------------------------------------
  it("shows the new label name in the affected range line", () => {
    const pending = pending_make({ label: "Communicate" });

    render(() => (
      <LabelOverwrite
        pending={pending}
        on_confirm={noop}
        on_keep_all={noop}
        on_cancel={noop}
      />
    ));

    expect(screen.getByText("Communicate")).toBeInTheDocument();
    expect(screen.getByText(/→/)).toBeInTheDocument();
  });

  // ------------------------------------------------------------------
  // Button labels
  // ------------------------------------------------------------------
  it("renders Overwrite All, Keep All, and Cancel buttons", () => {
    render(() => (
      <LabelOverwrite
        pending={pending_make()}
        on_confirm={noop}
        on_keep_all={noop}
        on_cancel={noop}
      />
    ));

    expect(screen.getByText("Overwrite All")).toBeInTheDocument();
    expect(screen.getByText("Keep All")).toBeInTheDocument();
    expect(screen.getByText("Cancel")).toBeInTheDocument();
  });

  // ------------------------------------------------------------------
  // Button callbacks
  // ------------------------------------------------------------------
  it("calls on_confirm when Overwrite All is clicked", () => {
    const on_confirm = vi.fn();
    render(() => (
      <LabelOverwrite
        pending={pending_make()}
        on_confirm={on_confirm}
        on_keep_all={noop}
        on_cancel={noop}
      />
    ));
    fireEvent.click(screen.getByText("Overwrite All"));
    expect(on_confirm).toHaveBeenCalledOnce();
  });

  it("calls on_keep_all when Keep All is clicked", () => {
    const on_keep_all = vi.fn();
    render(() => (
      <LabelOverwrite
        pending={pending_make()}
        on_confirm={noop}
        on_keep_all={on_keep_all}
        on_cancel={noop}
      />
    ));
    fireEvent.click(screen.getByText("Keep All"));
    expect(on_keep_all).toHaveBeenCalledOnce();
  });

  it("calls on_cancel when Cancel is clicked", () => {
    const on_cancel = vi.fn();
    render(() => (
      <LabelOverwrite
        pending={pending_make()}
        on_confirm={noop}
        on_keep_all={noop}
        on_cancel={on_cancel}
      />
    ));
    fireEvent.click(screen.getByText("Cancel"));
    expect(on_cancel).toHaveBeenCalledOnce();
  });
});
