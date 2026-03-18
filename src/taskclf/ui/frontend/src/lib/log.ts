import { host } from "./host";

const FRONTEND_LOG_MAX_LEN = 1000;

function debug_enabled(): boolean {
  const vite_meta = import.meta as ImportMeta & {
    env?: { DEV?: boolean };
  };
  return vite_meta.env?.DEV === true;
}

function arg_to_text(value: unknown): string {
  if (typeof value === "string") {
    return value;
  }
  if (value instanceof Error) {
    return value.stack ?? value.message;
  }
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
}

function payload_build(args: unknown[]): string {
  const raw = args.map((v) => arg_to_text(v)).join(" ");
  if (raw.length <= FRONTEND_LOG_MAX_LEN) {
    return raw;
  }
  return `${raw.slice(0, FRONTEND_LOG_MAX_LEN)}…`;
}

export function frontend_log_debug(...args: unknown[]): void {
  if (!debug_enabled()) {
    return;
  }
  console.debug(...args);
  if (!host.isNativeWindow) {
    return;
  }
  void host.invoke({
    cmd: "frontendDebugLog",
    message: payload_build(args),
  });
}

export function frontend_log_error(...args: unknown[]): void {
  console.error(...args);
  if (!debug_enabled()) {
    return;
  }
  if (!host.isNativeWindow) {
    return;
  }
  void host.invoke({
    cmd: "frontendErrorLog",
    message: payload_build(args),
  });
}

export function frontend_error_handlers_install(): () => void {
  if (!debug_enabled()) {
    return () => {};
  }

  const on_error = (event: ErrorEvent) => {
    frontend_log_error(
      "[window.onerror]",
      event.message,
      `${event.filename}:${event.lineno}:${event.colno}`,
      event.error,
    );
  };

  const on_unhandled_rejection = (event: PromiseRejectionEvent) => {
    frontend_log_error("[unhandledrejection]", event.reason);
  };

  window.addEventListener("error", on_error);
  window.addEventListener("unhandledrejection", on_unhandled_rejection);

  return () => {
    window.removeEventListener("error", on_error);
    window.removeEventListener("unhandledrejection", on_unhandled_rejection);
  };
}
