import type { PromptLabelEvent } from "./ws";

let permissionGranted = false;

export async function requestPermission(): Promise<boolean> {
  if (!("Notification" in window)) return false;
  if (Notification.permission === "granted") {
    permissionGranted = true;
    return true;
  }
  if (Notification.permission === "denied") return false;
  const result = await Notification.requestPermission();
  permissionGranted = result === "granted";
  return permissionGranted;
}

export function showTransitionNotification(
  prompt: PromptLabelEvent,
  onClick: () => void,
): Notification | null {
  if (!permissionGranted || !("Notification" in window)) return null;

  const body = prompt.suggested_label
    ? `${prompt.prev_app} → ${prompt.new_app} (${prompt.duration_min} min)\nSuggested: ${prompt.suggested_label}`
    : `${prompt.prev_app} → ${prompt.new_app} (${prompt.duration_min} min)`;

  const n = new Notification("taskclf — Activity changed", {
    body,
    tag: "taskclf-transition",
    renotify: true,
  });

  n.onclick = () => {
    window.focus();
    onClick();
    n.close();
  };

  return n;
}
