import {
  type Component,
  createEffect,
  createSignal,
  onCleanup,
  onMount,
  Show,
} from "solid-js";
import { LabelRecorder } from "./components/LabelRecorder";
import { LabelRecorderWindow } from "./components/LabelRecorderWindow";
import { PredictionBadge } from "./components/PredictionBadge";
import { StatusPanel } from "./components/StatusPanel";
import { StatusPanelWindow } from "./components/StatusPanelWindow";
import { host } from "./lib/host";
import { frontend_error_handlers_install } from "./lib/log";
import {
  notification_permission_ensure,
  transition_notification_show,
} from "./lib/notifications";
import { ws_store_new } from "./lib/ws";

/** Light page chrome for plain browser dev (Vite); contrasts with the dark frosted shell. */
const BROWSER_PAGE_BG = "#e8eaef";

const COMPACT_W = 150;
const CONTENT_W = 280;
const LABEL_MAX_H = 330;
const PANEL_MAX_H = 520;
const CHILD_HIDE_DELAY_MS = 300;

const view_param = new URLSearchParams(window.location.search).get("view");
const is_panel_view = view_param === "panel";
const is_label_view = view_param === "label";

if (host.kind === "browser") {
  document.documentElement.style.background = BROWSER_PAGE_BG;
  document.body.style.background = BROWSER_PAGE_BG;
} else {
  document.documentElement.style.background = "transparent";
  document.body.style.background = "transparent";
}

const App: Component = () => {
  if (is_label_view) {
    return <LabelRecorderWindow />;
  }
  if (is_panel_view) {
    return <StatusPanelWindow />;
  }

  const browser_compact = host.kind === "browser";
  const electron_shell = host.kind === "electron";
  const native_drag = host.kind !== "browser";
  const drag_spacer_class =
    native_drag && host.kind === "pywebview" ? "pywebview-drag-region" : undefined;
  const drag_spacer_style = {
    flex: "1",
    "min-width": "0",
    ...(electron_shell
      ? {
          "-webkit-app-region": "drag",
          "app-region": "drag",
        }
      : {}),
  };

  const ws = ws_store_new();

  const [label_pinned, set_label_pinned] = createSignal(false);
  const [badge_hovered, set_badge_hovered] = createSignal(false);
  const [label_hovered, set_label_hovered] = createSignal(false);
  const [panel_pinned, set_panel_pinned] = createSignal(false);
  const [dot_hovered, set_dot_hovered] = createSignal(false);
  const [panel_hovered, set_panel_hovered] = createSignal(false);
  const label_visible = () => label_pinned() || badge_hovered() || label_hovered();
  const panel_visible = () => panel_pinned() || dot_hovered() || panel_hovered();
  let label_hide_timer: ReturnType<typeof setTimeout> | null = null;
  let panel_hide_timer: ReturnType<typeof setTimeout> | null = null;

  const label_hide_cancel = () => {
    if (label_hide_timer !== null) {
      clearTimeout(label_hide_timer);
      label_hide_timer = null;
    }
  };

  const panel_hide_cancel = () => {
    if (panel_hide_timer !== null) {
      clearTimeout(panel_hide_timer);
      panel_hide_timer = null;
    }
  };

  const label_hide_schedule = () => {
    if (!browser_compact || label_pinned()) {
      return;
    }
    label_hide_cancel();
    label_hide_timer = setTimeout(() => {
      set_badge_hovered(false);
      set_label_hovered(false);
      label_hide_timer = null;
    }, CHILD_HIDE_DELAY_MS);
  };

  const panel_hide_schedule = () => {
    if (!browser_compact || panel_pinned()) {
      return;
    }
    panel_hide_cancel();
    panel_hide_timer = setTimeout(() => {
      set_dot_hovered(false);
      set_panel_hovered(false);
      panel_hide_timer = null;
    }, CHILD_HIDE_DELAY_MS);
  };

  const permission_ensure_once = (() => {
    let asked = false;
    return () => {
      if (!asked) {
        asked = true;
        notification_permission_ensure();
      }
    };
  })();

  onMount(() => {
    notification_permission_ensure();
    const cleanup_error_handlers = frontend_error_handlers_install();
    onCleanup(() => {
      label_hide_cancel();
      panel_hide_cancel();
      cleanup_error_handlers();
    });
  });

  createEffect(() => {
    const count = ws.label_grid_requested();
    if (count > 0) {
      if (browser_compact) {
        set_label_pinned(true);
      } else {
        host.invoke({ cmd: "toggleLabelGrid" });
      }
    }
  });

  createEffect(() => {
    const prompt = ws.latest_prompt();
    if (!prompt) {
      return;
    }
    transition_notification_show(prompt, () => {
      if (browser_compact) {
        set_label_pinned(true);
      } else {
        host.invoke({ cmd: "toggleLabelGrid" });
      }
    });
  });

  return (
    <div
      style={{
        display: "flex",
        "flex-direction": "column",
        "align-items": browser_compact ? "flex-end" : "stretch",
        ...(browser_compact
          ? {
              "min-height": "100vh",
              "box-sizing": "border-box",
              padding: "16px",
              background: BROWSER_PAGE_BG,
            }
          : {}),
      }}
    >
      <div
        style={{
          display: "flex",
          "flex-direction": "column",
          "align-items": browser_compact ? "flex-end" : "stretch",
          width: browser_compact ? "auto" : undefined,
          "max-width": "100%",
        }}
      >
        <div
          style={{
            background: "rgba(15, 17, 23, 0.5)",
            "backdrop-filter": "blur(20px)",
            "-webkit-backdrop-filter": "blur(20px)",
            width: browser_compact ? `${COMPACT_W}px` : "100%",
            ...(browser_compact
              ? { "box-shadow": "0 4px 24px rgba(0, 0, 0, 0.5)" }
              : { height: "100vh" }),
            overflow: "hidden",
            "border-radius": "20px",
            display: "flex",
            "flex-direction": "column",
            "min-height": "0",
          }}
        >
          <div
            style={{
              display: "flex",
              "flex-direction": "column",
              "min-height": "0",
              flex: browser_compact ? undefined : "1",
            }}
          >
            <div
              style={{
                display: "flex",
                "align-items": "stretch",
                padding: "0 8px 0 12px",
                height: "30px",
                gap: "6px",
                "user-select": "none",
                "flex-shrink": "0",
              }}
            >
              <div
                class={drag_spacer_class}
                style={drag_spacer_style}
                aria-hidden="true"
              />
              <div
                style={{
                  flex: "0 0 auto",
                  display: "flex",
                  "align-items": "center",
                  "justify-content": "center",
                }}
              >
                <PredictionBadge
                  status={ws.connection_status}
                  latest_status={ws.latest_status}
                  latest_prediction={ws.latest_prediction}
                  latest_tray_state={ws.latest_tray_state}
                  active_suggestion={ws.active_suggestion}
                  label_pinned={label_pinned}
                  panel_pinned={panel_pinned}
                  on_toggle_panel={
                    browser_compact
                      ? () => {
                          permission_ensure_once();
                          if (panel_visible() && panel_pinned()) {
                            panel_hide_cancel();
                            set_panel_pinned(false);
                            set_dot_hovered(false);
                            set_panel_hovered(false);
                          } else if (panel_visible()) {
                            panel_hide_cancel();
                            set_panel_pinned(true);
                          } else {
                            panel_hide_cancel();
                            set_panel_pinned(true);
                          }
                        }
                      : () => {
                          permission_ensure_once();
                          host.invoke({ cmd: "toggleStatePanel" });
                        }
                  }
                  on_show_panel={
                    browser_compact
                      ? () => {
                          permission_ensure_once();
                          panel_hide_cancel();
                          set_dot_hovered(true);
                        }
                      : () => {
                          permission_ensure_once();
                          host.invoke({ cmd: "showStatePanel" });
                        }
                  }
                  on_hide_panel={
                    browser_compact
                      ? panel_hide_schedule
                      : () => {
                          host.invoke({ cmd: "hideStatePanel" });
                        }
                  }
                  on_toggle_label={
                    browser_compact
                      ? () => {
                          permission_ensure_once();
                          if (label_visible() && label_pinned()) {
                            label_hide_cancel();
                            set_label_pinned(false);
                            set_badge_hovered(false);
                            set_label_hovered(false);
                          } else if (label_visible()) {
                            label_hide_cancel();
                            set_label_pinned(true);
                          } else {
                            label_hide_cancel();
                            set_label_pinned(true);
                          }
                        }
                      : () => {
                          permission_ensure_once();
                          host.invoke({ cmd: "toggleLabelGrid" });
                        }
                  }
                  on_show_label={
                    browser_compact
                      ? () => {
                          label_hide_cancel();
                          set_badge_hovered(true);
                        }
                      : () => {
                          host.invoke({ cmd: "showLabelGrid" });
                        }
                  }
                  on_hide_label={
                    browser_compact
                      ? label_hide_schedule
                      : () => {
                          host.invoke({ cmd: "hideLabelGrid" });
                        }
                  }
                />
              </div>
              <div
                class={drag_spacer_class}
                style={drag_spacer_style}
                aria-hidden="true"
              />
            </div>
          </div>
        </div>

        <Show when={browser_compact && label_visible()}>
          {/* biome-ignore lint/a11y/noStaticElementInteractions: hover bridge for label popup */}
          <div
            onMouseEnter={() => {
              label_hide_cancel();
              set_label_hovered(true);
            }}
            onMouseLeave={() => {
              set_label_hovered(false);
              label_hide_schedule();
            }}
            style={{
              width: `${CONTENT_W}px`,
              "max-height": `${LABEL_MAX_H}px`,
              "overflow-y": "auto",
              "margin-top": "4px",
              background: "var(--bg)",
              "border-radius": "12px",
              "box-sizing": "border-box",
            }}
          >
            <LabelRecorder
              on_collapse={() => {
                label_hide_cancel();
                set_label_pinned(false);
                set_badge_hovered(false);
                set_label_hovered(false);
              }}
              prediction={ws.latest_prediction}
              suggestion={ws.active_suggestion}
              on_suggestion_dismiss={ws.suggestion_dismiss}
            />
          </div>
        </Show>

        <Show when={browser_compact && panel_visible()}>
          {/* biome-ignore lint/a11y/noStaticElementInteractions: hover bridge for panel */}
          <div
            onMouseEnter={() => {
              panel_hide_cancel();
              set_panel_hovered(true);
            }}
            onMouseLeave={() => {
              set_panel_hovered(false);
              panel_hide_schedule();
            }}
            style={{
              width: `${CONTENT_W}px`,
              "max-height": `${PANEL_MAX_H}px`,
              "overflow-y": "auto",
              "margin-top": "4px",
            }}
          >
            <StatusPanel
              status={ws.connection_status}
              latest_status={ws.latest_status}
              latest_prediction={ws.latest_prediction}
              latest_tray_state={ws.latest_tray_state}
              active_suggestion={ws.active_suggestion}
              ws_stats={ws.ws_stats}
              train_state={ws.train_state}
              on_open_label_recorder={() => {
                permission_ensure_once();
                label_hide_cancel();
                set_label_pinned(true);
                set_badge_hovered(true);
                set_label_hovered(true);
              }}
            />
          </div>
        </Show>
      </div>
    </div>
  );
};

export default App;
