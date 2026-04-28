import { type Accessor, createEffect, onMount } from "solid-js";
import { host } from "./host";
import {
  notification_permission_ensure,
  transition_notification_show,
} from "./notifications";
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

const LAST_PROMPT_NOTIFICATION_KEY = "taskclf:lastTransitionPromptNotification";

function transition_prompt_notification_key(prompt: PromptLabelEvent): string {
  return [
    prompt.block_start,
    prompt.block_end,
    prompt.prev_app,
    prompt.new_app,
    prompt.suggested_label ?? "",
  ].join("|");
}

function transition_prompt_notification_claim(prompt: PromptLabelEvent): boolean {
  const key = transition_prompt_notification_key(prompt);
  try {
    if (window.localStorage.getItem(LAST_PROMPT_NOTIFICATION_KEY) === key) {
      return false;
    }
    window.localStorage.setItem(LAST_PROMPT_NOTIFICATION_KEY, key);
  } catch {
    // localStorage can be unavailable in hardened browser contexts; fall through.
  }
  return true;
}

export function transition_prompt_notifications_bind(
  prompt: Accessor<PromptLabelEvent | null>,
  on_open_label_grid: () => void,
): void {
  onMount(() => {
    if (host.kind === "browser") {
      void notification_permission_ensure();
    }
  });

  createEffect(() => {
    const next_prompt = prompt();
    if (!next_prompt) {
      return;
    }

    const notification_key = transition_prompt_notification_key(next_prompt);
    const claimed = transition_prompt_notification_claim(next_prompt);
    // #region agent log
    agent_debug_log(
      "pre-fix",
      "H1,H2,H4",
      "src/taskclf/ui/frontend/src/lib/transitionPromptNotifications.ts:45",
      "transition prompt notification effect evaluated",
      {
        claimed,
        notification_key,
        host_kind: host.kind,
        href: window.location.href,
        suggested_label: next_prompt.suggested_label,
        suggestion_text_present: next_prompt.suggestion_text != null,
      },
    );
    // #endregion
    if (!claimed) {
      return;
    }

    if (host.kind !== "browser") {
      // #region agent log
      agent_debug_log(
        "pre-fix",
        "H4",
        "src/taskclf/ui/frontend/src/lib/transitionPromptNotifications.ts:50",
        "routing prompt notification through native host",
        {
          host_kind: host.kind,
          href: window.location.href,
          notification_key,
        },
      );
      // #endregion
      void host.invoke({ cmd: "showTransitionNotification", prompt: next_prompt });
      return;
    }

    // #region agent log
    agent_debug_log(
      "pre-fix",
      "H3",
      "src/taskclf/ui/frontend/src/lib/transitionPromptNotifications.ts:55",
      "routing prompt notification through browser api",
      {
        notification_key,
        notification_permission:
          "Notification" in window ? Notification.permission : "unavailable",
        href: window.location.href,
      },
    );
    // #endregion
    transition_notification_show(next_prompt, on_open_label_grid);
  });
}
