/**
 * Normalize wire-format JSON (`null`) into app-layer optional values (`undefined`).
 * Use at fetch / WebSocket parse boundaries only.
 */
export function null_to_undefined<T>(value: T): T {
  if (value === null) {
    return undefined as T;
  }
  if (Array.isArray(value)) {
    return value.map((item) => null_to_undefined(item)) as T;
  }
  if (typeof value === "object") {
    const record = value as Record<string, unknown>;
    const out: Record<string, unknown> = {};
    for (const key of Object.keys(record)) {
      const v = record[key];
      if (v !== null) {
        out[key] = null_to_undefined(v);
      }
    }
    return out as T;
  }
  return value;
}
