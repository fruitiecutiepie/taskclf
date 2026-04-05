import { frontend_log_debug } from "./log";
import type { PromptLabelEvent } from "./ws";

// `renotify` is part of the Web Notifications API spec but missing from
// TypeScript's lib.dom.d.ts. Extend until upstream adds it.
// https://developer.mozilla.org/en-US/docs/Web/API/Notification/Notification#renotify
type NotificationOptionsExtended = NotificationOptions & {
  renotify?: boolean;
  requireInteraction?: boolean;
};

let permission_granted = false;

function notification_range_format(prompt: PromptLabelEvent): string {
  const start = new Date(prompt.block_start);
  const end = new Date(prompt.block_end);
  if (Number.isNaN(start.getTime()) || Number.isNaN(end.getTime())) {
    return `${prompt.block_start} → ${prompt.block_end}`;
  }

  const same_local_day =
    start.getFullYear() === end.getFullYear()
    && start.getMonth() === end.getMonth()
    && start.getDate() === end.getDate();

  if (same_local_day) {
    return `${start.toLocaleTimeString([], {
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    })} → ${end.toLocaleTimeString([], {
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    })}`;
  }

  return `${start.toLocaleString([], {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  })} → ${end.toLocaleString([], {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  })}`;
}

export async function notification_permission_ensure(): Promise<boolean> {
  if (!("Notification" in window)) {
    frontend_log_debug("[notifications] Notification API not available in window");
    return false;
  }
  frontend_log_debug("[notifications] Current permission:", Notification.permission);
  if (Notification.permission === "granted") {
    permission_granted = true;
    frontend_log_debug("[notifications] Permission already granted");
    return true;
  }
  if (Notification.permission === "denied") {
    frontend_log_debug("[notifications] Permission denied");
    return false;
  }
  frontend_log_debug("[notifications] Requesting permission...");
  const result = await Notification.requestPermission();
  permission_granted = result === "granted";
  frontend_log_debug("[notifications] Permission request result:", result);
  return permission_granted;
}

export function transition_notification_show(
  prompt: PromptLabelEvent,
  on_click: () => void,
): Notification | null {
  if (!permission_granted || !("Notification" in window)) {
    return null;
  }

  const range = notification_range_format(prompt);
  const body = prompt.suggestion_text
    ? `${prompt.suggestion_text}\n${range}`
    : `${prompt.prev_app} → ${prompt.new_app}\n${range}`;

  const n = new Notification("taskclf — Activity changed", {
    body,
    tag: "taskclf-transition",
    renotify: true,
    // Keep transition prompts visible until the user acts on supported runtimes.
    requireInteraction: true,
  } satisfies NotificationOptionsExtended as NotificationOptions);

  n.onclick = () => {
    window.focus();
    on_click();
    n.close();
  };

  return n;
}
