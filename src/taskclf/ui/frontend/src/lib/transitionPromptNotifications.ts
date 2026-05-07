import { type Accessor, createEffect, onMount } from "solid-js";
import { host } from "./host";
import {
  notification_permission_ensure,
  transition_notification_show,
} from "./notifications";
import type { PromptLabelEvent } from "./ws";

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

function transition_prompt_clone(prompt: PromptLabelEvent): PromptLabelEvent {
  return {
    type: "prompt_label",
    prev_app: prompt.prev_app,
    new_app: prompt.new_app,
    block_start: prompt.block_start,
    block_end: prompt.block_end,
    duration_min: prompt.duration_min,
    suggested_label: prompt.suggested_label,
    suggestion_text: prompt.suggestion_text,
  };
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

    const claimed = transition_prompt_notification_claim(next_prompt);
    if (!claimed) {
      return;
    }

    const cloneable_prompt = transition_prompt_clone(next_prompt);

    if (host.kind !== "browser") {
      void host.invoke({
        cmd: "showTransitionNotification",
        prompt: cloneable_prompt,
      });
      return;
    }

    transition_notification_show(cloneable_prompt, on_open_label_grid);
  });
}
