import type { OverwritePending } from "../components/LabelOverwrite";

export type OverwritePendingParams = {
  label: string;
  start: string;
  end: string;
  confidence: number;
  extend_forward: boolean;
};

/** Parse structured 409 overlap detail from `api_json` error messages (`409: {...}`). */
export function overwrite_pending_from_api_error(
  err: unknown,
  params: OverwritePendingParams,
): OverwritePending | null {
  const msg = err instanceof Error ? err.message : "";
  const json_match = msg.match(/\{[\s\S]*\}/);
  if (!json_match) {
    return null;
  }
  try {
    const parsed = JSON.parse(json_match[0]);
    const overlap = parsed.detail ?? parsed;
    const spans: { start_ts: string; end_ts: string; label: string }[] =
      overlap.conflicting_spans ?? [];
    if (
      spans.length === 0
      && overlap.conflicting_start_ts
      && overlap.conflicting_end_ts
    ) {
      spans.push({
        start_ts: overlap.conflicting_start_ts,
        end_ts: overlap.conflicting_end_ts,
        label: overlap.conflicting_label ?? "unknown",
      });
    }
    if (spans.length === 0) {
      return null;
    }
    return {
      label: params.label,
      start: params.start,
      end: params.end,
      conflicts: spans,
      confidence: params.confidence,
      extend_forward: params.extend_forward,
    };
  } catch {
    return null;
  }
}
