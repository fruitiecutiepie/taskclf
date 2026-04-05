import { render, screen } from "@solidjs/testing-library";
import { describe, expect, it, vi } from "vitest";
import { time_format } from "../lib/format";
import type { LabelSuggestion } from "../lib/ws";
import { PredictionSuggestion } from "./PredictionSuggestion";

vi.mock("../lib/api", () => ({
  notification_accept: vi.fn(),
  notification_skip: vi.fn(),
}));

vi.mock("../lib/log", () => ({
  frontend_log_error: vi.fn(),
}));

function suggestion_make(overrides: Partial<LabelSuggestion> = {}): LabelSuggestion {
  return {
    type: "suggest_label",
    reason: "App transition suggested a new label",
    old_label: "ReadResearch",
    suggested: "Write",
    confidence: 0.92,
    block_start: "2026-04-05T12:00:00Z",
    block_end: "2026-04-05T13:00:00Z",
    ...overrides,
  };
}

function suggestion_range_text(block_start: string, block_end: string): string {
  const start = new Date(block_start);
  const end = new Date(block_end);
  const crosses_local_day =
    start.getFullYear() !== end.getFullYear()
    || start.getMonth() !== end.getMonth()
    || start.getDate() !== end.getDate();

  if (!crosses_local_day) {
    return `Applies to ${time_format(block_start)} → ${time_format(block_end)}`;
  }

  return `Applies to ${start.toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  })} → ${end.toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  })}`;
}

describe("PredictionSuggestion", () => {
  it("shows the applicable suggestion time range", () => {
    const suggestion = suggestion_make();

    render(() => <PredictionSuggestion suggestion={() => suggestion} />);

    expect(
      screen.getByText(
        suggestion_range_text(suggestion.block_start, suggestion.block_end),
      ),
    ).toBeInTheDocument();
  });

  it("includes dates when the suggestion spans local midnight", () => {
    const suggestion = suggestion_make({
      block_start: new Date(2026, 3, 5, 23, 50, 0).toISOString(),
      block_end: new Date(2026, 3, 6, 0, 10, 0).toISOString(),
    });

    render(() => <PredictionSuggestion suggestion={() => suggestion} />);

    expect(
      screen.getByText(
        suggestion_range_text(suggestion.block_start, suggestion.block_end),
      ),
    ).toBeInTheDocument();
  });
});
