export function parseDate(iso: string): Date {
  return new Date(iso);
}

export function parseISODate(iso: string): Date {
  return new Date(iso);
}

export function todayDateStr(): string {
  const d = new Date();
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;
}

export function fmtDateLabel(dateStr: string): string {
  const today = todayDateStr();
  if (dateStr === today) {
    return "Today";
  }
  const yesterday = new Date();
  yesterday.setDate(yesterday.getDate() - 1);
  const yStr = `${yesterday.getFullYear()}-${String(yesterday.getMonth() + 1).padStart(2, "0")}-${String(yesterday.getDate()).padStart(2, "0")}`;
  if (dateStr === yStr) {
    return "Yesterday";
  }
  const d = new Date(`${dateStr}T00:00:00`);
  return d.toLocaleDateString([], { weekday: "short", month: "short", day: "numeric" });
}

export function shiftDate(dateStr: string, delta: number): string {
  const d = new Date(`${dateStr}T12:00:00`);
  d.setDate(d.getDate() + delta);
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;
}

export function fmtTime(d: Date): string {
  return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

export function fmtTimeSec(d: Date): string {
  return d.toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

export function fmtDuration(ms: number): string {
  const totalMin = Math.round(ms / 60_000);
  if (totalMin < 1) {
    return "<1m";
  }
  const h = Math.floor(totalMin / 60);
  const m = totalMin % 60;
  if (h === 0) {
    return `${m}m`;
  }
  if (m === 0) {
    return `${h}h`;
  }
  return `${h}h ${m}m`;
}

export function toTimeInputValue(d: Date): string {
  return `${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}`;
}

export function toTimeInputValueSec(d: Date): string {
  return `${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}:${String(d.getSeconds()).padStart(2, "0")}`;
}

export function timeInputToDate(dateStr: string, timeVal: string): Date {
  const parts = timeVal.split(":");
  const hh = parts[0] ?? "00";
  const mm = parts[1] ?? "00";
  const ss = parts[2] ?? "00";
  return new Date(`${dateStr}T${hh}:${mm}:${ss}`);
}

export function timeAgo(iso: string): string {
  const d = parseISODate(iso);
  const mins = Math.round((Date.now() - d.getTime()) / 60_000);
  if (mins < 1) {
    return "just now";
  }
  if (mins < 60) {
    return `${mins}m ago`;
  }
  const h = Math.floor(mins / 60);
  if (h < 24) {
    return `${h}h ago`;
  }
  const days = Math.floor(h / 24);
  return `${days}d ago`;
}

export type TimeRange = {
  start: string;
  end: string;
};

export function timeRangeForMinutes(mins: number): TimeRange | null {
  if (mins < 1) {
    return null;
  }
  const now = new Date();
  const start = new Date(now.getTime() - mins * 60_000);
  return {
    start: start.toISOString(),
    end: now.toISOString(),
  };
}
