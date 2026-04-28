import { frontend_log_debug } from "./log";
import type { PromptLabelEvent } from "./ws";

// #region agent log
function agent_debug_log(
  runId: string,
  hypothesisId: string,
  location: string,
  message: string,
  data: Record<string, unknown>,
) {
  fetch("http://localhost:7434/ingest/307992f9-e352-421f-9c8b-95a59cddc80f", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-Debug-Session-Id": "f37ed4",
    },
    body: JSON.stringify({
      sessionId: "f37ed4",
      runId,
      hypothesisId,
      location,
      message,
      data,
      timestamp: Date.now(),
    }),
  }).catch(() => {});
}
// #endregion

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
  // #region agent log
  agent_debug_log(
    "pre-fix",
    "H3",
    "src/taskclf/ui/frontend/src/lib/notifications.ts:53",
    "notification permission ensure entered",
    {
      notification_api_available: "Notification" in window,
      permission: "Notification" in window ? Notification.permission : "unavailable",
      permission_granted_cache: permission_granted,
      href: window.location.href,
    },
  );
  // #endregion
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
  // #region agent log
  agent_debug_log(
    "pre-fix",
    "H3",
    "src/taskclf/ui/frontend/src/lib/notifications.ts:69",
    "notification permission request resolved",
    {
      result,
      permission_granted_cache: permission_granted,
      href: window.location.href,
    },
  );
  // #endregion
  frontend_log_debug("[notifications] Permission request result:", result);
  return permission_granted;
}

export function transition_notification_show(
  prompt: PromptLabelEvent,
  on_click: () => void,
): Notification | null {
  // #region agent log
  agent_debug_log(
    "pre-fix",
    "H3",
    "src/taskclf/ui/frontend/src/lib/notifications.ts:75",
    "browser transition notification show entered",
    {
      permission_granted_cache: permission_granted,
      notification_api_available: "Notification" in window,
      permission: "Notification" in window ? Notification.permission : "unavailable",
      block_start: prompt.block_start,
      block_end: prompt.block_end,
      suggested_label: prompt.suggested_label,
      href: window.location.href,
    },
  );
  // #endregion
  if (!permission_granted || !("Notification" in window)) {
    // #region agent log
    agent_debug_log(
      "pre-fix",
      "H3",
      "src/taskclf/ui/frontend/src/lib/notifications.ts:79",
      "browser transition notification skipped",
      {
        permission_granted_cache: permission_granted,
        notification_api_available: "Notification" in window,
        permission: "Notification" in window ? Notification.permission : "unavailable",
        href: window.location.href,
      },
    );
    // #endregion
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
  // #region agent log
  agent_debug_log(
    "pre-fix",
    "H3",
    "src/taskclf/ui/frontend/src/lib/notifications.ts:88",
    "browser transition notification constructed",
    {
      title: "taskclf — Activity changed",
      body,
      tag: "taskclf-transition",
      href: window.location.href,
    },
  );
  // #endregion

  n.onclick = () => {
    window.focus();
    on_click();
    n.close();
  };

  return n;
}
