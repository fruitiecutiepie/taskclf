export function date_parse(iso: string): Date {
  return new Date(iso);
}

export function iso_date_parse(iso: string): Date {
  return new Date(iso);
}

export function date_today_str(): string {
  const d = new Date();
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;
}

export function date_label_fmt(dateStr: string): string {
  const today = date_today_str();
  if (dateStr === today) {
    return "Today";
  }
  const yesterday = new Date();
  yesterday.setDate(yesterday.getDate() - 1);
  const y_str = `${yesterday.getFullYear()}-${String(yesterday.getMonth() + 1).padStart(2, "0")}-${String(yesterday.getDate()).padStart(2, "0")}`;
  if (dateStr === y_str) {
    return "Yesterday";
  }
  const d = new Date(`${dateStr}T00:00:00`);
  return d.toLocaleDateString([], { weekday: "short", month: "short", day: "numeric" });
}

export function date_shift(dateStr: string, delta: number): string {
  const d = new Date(`${dateStr}T12:00:00`);
  d.setDate(d.getDate() + delta);
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;
}

export function time_fmt(d: Date): string {
  return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

export function time_sec_fmt(d: Date): string {
  return d.toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

export function time_fmt_conditional_sec(d: Date): string {
  if (d.getSeconds() !== 0) {
    return time_sec_fmt(d);
  }
  return time_fmt(d);
}

export function duration_fmt(ms: number): string {
  const total_min = Math.round(ms / 60_000);
  if (total_min < 1) {
    return "<1m";
  }
  const h = Math.floor(total_min / 60);
  const m = total_min % 60;
  if (h === 0) {
    return `${m}m`;
  }
  if (m === 0) {
    return `${h}h`;
  }
  return `${h}h ${m}m`;
}

export function time_input_value(d: Date): string {
  return `${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}`;
}

export function time_input_value_sec(d: Date): string {
  return `${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}:${String(d.getSeconds()).padStart(2, "0")}`;
}

export function time_input_date(dateStr: string, timeVal: string): Date {
  const parts = timeVal.split(":");
  const hh = parts[0] ?? "00";
  const mm = parts[1] ?? "00";
  const ss = parts[2] ?? "00";
  return new Date(`${dateStr}T${hh}:${mm}:${ss}`);
}

/**
 * Quick-label gap shortcut text from a label's end time, or `null` when the
 * control should stay hidden (under one rounded minute since end).
 */
export function gap_shortcut_label_from_end(
  end_ms: number,
  now_ms: number,
): string | null {
  const ago = Math.round((now_ms - end_ms) / 60_000);
  if (ago < 1) {
    return null;
  }
  if (ago >= 60) {
    return `gap ${Math.floor(ago / 60)}h${ago % 60 ? `${ago % 60}m` : ""}`;
  }
  return `gap ${ago}m`;
}

export function time_ago(iso: string): string {
  const d = iso_date_parse(iso);
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

export function time_range_minutes(mins: number): TimeRange | null {
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
