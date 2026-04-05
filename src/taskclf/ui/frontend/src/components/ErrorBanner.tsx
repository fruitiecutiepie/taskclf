import { type Component, createSignal, onCleanup, Show } from "solid-js";
import { frontend_log_error } from "../lib/log";

async function error_text_copy(text: string): Promise<void> {
  if (typeof navigator !== "undefined" && navigator.clipboard?.writeText) {
    await navigator.clipboard.writeText(text);
    return;
  }

  if (typeof document === "undefined") {
    throw new Error("Clipboard unavailable");
  }

  const textarea = document.createElement("textarea");
  textarea.value = text;
  textarea.setAttribute("readonly", "");
  textarea.style.position = "fixed";
  textarea.style.opacity = "0";
  textarea.style.pointerEvents = "none";
  textarea.style.left = "-9999px";
  document.body.append(textarea);
  textarea.focus();
  textarea.select();

  const copied = document.execCommand("copy");
  textarea.remove();
  if (!copied) {
    throw new Error("Clipboard unavailable");
  }
}

export const ErrorBanner: Component<{
  message: string;
  on_close?: () => void;
}> = (props) => {
  const [copy_state, set_copy_state] = createSignal<"idle" | "copied" | "failed">(
    "idle",
  );
  let reset_timer: ReturnType<typeof setTimeout> | null = null;

  onCleanup(() => {
    if (reset_timer !== null) {
      clearTimeout(reset_timer);
    }
  });

  async function error_copy() {
    try {
      await error_text_copy(props.message);
      set_copy_state("copied");
    } catch (err: unknown) {
      frontend_log_error("Failed to copy error message", err);
      set_copy_state("failed");
    } finally {
      if (reset_timer !== null) {
        clearTimeout(reset_timer);
      }
      reset_timer = setTimeout(() => {
        set_copy_state("idle");
        reset_timer = null;
      }, 2000);
    }
  }

  const copy_label = () => {
    if (copy_state() === "copied") {
      return "Copied";
    }
    if (copy_state() === "failed") {
      return "Copy failed";
    }
    return "Copy error";
  };

  const action_style = {
    background: "none",
    border: "none",
    color: "inherit",
    cursor: "pointer",
    "font-size": "0.58rem",
    "font-family": "inherit",
    padding: "0",
    "line-height": "1.3",
  };

  return (
    <div
      role="alert"
      aria-live="assertive"
      style={{
        display: "flex",
        "justify-content": "space-between",
        "align-items": "flex-start",
        gap: "8px",
        width: "100%",
        "box-sizing": "border-box",
        padding: "6px 8px",
        "margin-top": "6px",
        "border-radius": "6px",
        border: "1px solid rgba(239,68,68,0.28)",
        background: "rgba(239,68,68,0.08)",
        color: "#ef4444",
        "font-size": "0.58rem",
      }}
    >
      <div style={{ "min-width": "0", flex: "1 1 auto" }}>
        <strong style={{ "margin-right": "4px" }}>Error:</strong>
        <span>{props.message}</span>
      </div>
      <div
        style={{
          display: "flex",
          gap: "8px",
          "align-items": "center",
          "flex-shrink": "0",
          "white-space": "nowrap",
        }}
      >
        <button type="button" onClick={error_copy} style={action_style}>
          {copy_label()}
        </button>
        <Show when={props.on_close}>
          <button
            type="button"
            aria-label="Close error"
            onClick={() => props.on_close?.()}
            style={action_style}
          >
            Close
          </button>
        </Show>
      </div>
    </div>
  );
};
