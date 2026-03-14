export function formatDuration(seconds: number): string {
  if (seconds < 60) return `${seconds}s`;
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  if (m < 60) return s > 0 ? `${m}m ${s}s` : `${m}m`;
  const h = Math.floor(m / 60);
  const rm = m % 60;
  return rm > 0 ? `${h}h ${rm}m` : `${h}h`;
}

export function formatTime(iso: string | null | undefined): string {
  if (!iso) return "—";
  try {
    const d = new Date(iso);
    return d.toLocaleTimeString([], {
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    });
  } catch {
    return iso;
  }
}

export function truncPath(p: string | null | undefined, maxLen = 30): string {
  if (!p) return "—";
  if (p.length <= maxLen) return p;
  return `…${p.slice(-(maxLen - 1))}`;
}

export function shortAppName(app: string): string {
  const parts = app.split(".");
  return parts[parts.length - 1];
}

export function fmtRate(v: number | null): string | null {
  if (v == null) return null;
  return v < 10 ? v.toFixed(1) : String(Math.round(v));
}
