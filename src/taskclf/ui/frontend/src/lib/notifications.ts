import type { PromptLabelEvent } from "./ws";

// `renotify` is part of the Web Notifications API spec but missing from
// TypeScript's lib.dom.d.ts. Extend until upstream adds it.
// https://developer.mozilla.org/en-US/docs/Web/API/Notification/Notification#renotify
type NotificationOptionsExtended = NotificationOptions & {
  renotify?: boolean;
};

let permissionGranted = false;

export async function notification_permission_ensure(): Promise<boolean> {
  if (!("Notification" in window)) {
    return false;
  }
  if (Notification.permission === "granted") {
    permissionGranted = true;
    return true;
  }
  if (Notification.permission === "denied") {
    return false;
  }
  const result = await Notification.requestPermission();
  permissionGranted = result === "granted";
  return permissionGranted;
}

export function transition_notification_show(
  prompt: PromptLabelEvent,
  onClick: () => void,
): Notification | null {
  if (!permissionGranted || !("Notification" in window)) {
    return null;
  }

  const body = prompt.suggested_label
    ? `${prompt.prev_app} → ${prompt.new_app} (${prompt.duration_min} min)\nSuggested: ${prompt.suggested_label}`
    : `${prompt.prev_app} → ${prompt.new_app} (${prompt.duration_min} min)`;

  const n = new Notification("taskclf — Activity changed", {
    body,

    tag: "taskclf-transition",
    renotify: true,
  } satisfies NotificationOptionsExtended as NotificationOptions);

  n.onclick = () => {
    window.focus();
    onClick();
    n.close();
  };

  return n;
}
