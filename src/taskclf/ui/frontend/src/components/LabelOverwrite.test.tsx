import { fireEvent, render, screen } from "@solidjs/testing-library";
import { describe, expect, it, vi } from "vitest";
import { LabelOverwrite, type OverwritePending } from "./LabelOverwrite";

function makePending(overrides: Partial<OverwritePending> = {}): OverwritePending {
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
    extendForward: false,
    ...overrides,
  };
}

const noop = () => {};

describe("LabelOverwrite", () => {
  // ------------------------------------------------------------------
  // Sorting
  // ------------------------------------------------------------------
  it("renders conflicts sorted by start_ts", () => {
    const pending = makePending({
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
        onConfirm={noop}
        onKeepAll={noop}
        onCancel={noop}
      />
    ));

    fireEvent.click(screen.getByText("show details"));

    const allLabels = screen.getAllByText(/Review|Build|Debug/);
    // First 3 are the summary (sorted by first appearance), last 3 are detail rows (sorted by start_ts)
    const detailLabels = allLabels.slice(3);
    const detailOrder = detailLabels.map((el) => el.textContent);
    expect(detailOrder).toEqual(["Review", "Build", "Debug"]);
  });

  // ------------------------------------------------------------------
  // Single conflict: shows detail inline, no expand toggle
  // ------------------------------------------------------------------
  it("shows span detail inline for a single conflict", () => {
    const pending = makePending();

    render(() => (
      <LabelOverwrite
        pending={pending}
        onConfirm={noop}
        onKeepAll={noop}
        onCancel={noop}
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
    const pending = makePending({
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
        onConfirm={noop}
        onKeepAll={noop}
        onCancel={noop}
      />
    ));

    expect(screen.getByText(/2 labels:/)).toBeInTheDocument();
    expect(screen.getByText("show details")).toBeInTheDocument();
  });

  it("toggles detail visibility on click", () => {
    const pending = makePending({
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
        onConfirm={noop}
        onKeepAll={noop}
        onCancel={noop}
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
    const pending = makePending({
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
        onConfirm={noop}
        onKeepAll={noop}
        onCancel={noop}
      />
    ));

    const summaryRow = screen.getByText(/3 labels:/);
    expect(summaryRow.parentElement?.textContent).toContain("Review");
    expect(summaryRow.parentElement?.textContent).toContain("Debug");
    const summaryText = summaryRow.parentElement?.textContent ?? "";
    const reviewCount = summaryText.split("Review").length - 1;
    expect(reviewCount).toBe(1);
  });

  // ------------------------------------------------------------------
  // Shows new label name in affected range
  // ------------------------------------------------------------------
  it("shows the new label name in the affected range line", () => {
    const pending = makePending({ label: "Communicate" });

    render(() => (
      <LabelOverwrite
        pending={pending}
        onConfirm={noop}
        onKeepAll={noop}
        onCancel={noop}
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
        pending={makePending()}
        onConfirm={noop}
        onKeepAll={noop}
        onCancel={noop}
      />
    ));

    expect(screen.getByText("Overwrite All")).toBeInTheDocument();
    expect(screen.getByText("Keep All")).toBeInTheDocument();
    expect(screen.getByText("Cancel")).toBeInTheDocument();
  });

  // ------------------------------------------------------------------
  // Button callbacks
  // ------------------------------------------------------------------
  it("calls onConfirm when Overwrite All is clicked", () => {
    const onConfirm = vi.fn();
    render(() => (
      <LabelOverwrite
        pending={makePending()}
        onConfirm={onConfirm}
        onKeepAll={noop}
        onCancel={noop}
      />
    ));
    fireEvent.click(screen.getByText("Overwrite All"));
    expect(onConfirm).toHaveBeenCalledOnce();
  });

  it("calls onKeepAll when Keep All is clicked", () => {
    const onKeepAll = vi.fn();
    render(() => (
      <LabelOverwrite
        pending={makePending()}
        onConfirm={noop}
        onKeepAll={onKeepAll}
        onCancel={noop}
      />
    ));
    fireEvent.click(screen.getByText("Keep All"));
    expect(onKeepAll).toHaveBeenCalledOnce();
  });

  it("calls onCancel when Cancel is clicked", () => {
    const onCancel = vi.fn();
    render(() => (
      <LabelOverwrite
        pending={makePending()}
        onConfirm={noop}
        onKeepAll={noop}
        onCancel={onCancel}
      />
    ));
    fireEvent.click(screen.getByText("Cancel"));
    expect(onCancel).toHaveBeenCalledOnce();
  });
});
