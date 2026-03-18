import { type Component, createEffect, createSignal, onMount, Show } from "solid-js";
import { LabelRecorder } from "./components/LabelRecorder";
import { LabelRecorderWindow } from "./components/LabelRecorderWindow";
import { PredictionBadge } from "./components/PredictionBadge";
import { StatusPanel } from "./components/StatusPanel";
import { StatusPanelWindow } from "./components/StatusPanelWindow";
import { host } from "./lib/host";
import {
  notification_permission_ensure,
  transition_notification_show,
} from "./lib/notifications";
import { ws_store_new } from "./lib/ws";

const view_param = new URLSearchParams(window.location.search).get("view");
const is_panel_view = view_param === "panel";
const is_label_view = view_param === "label";

const COMPACT_W = 150;
const CONTENT_W = 280;
const LABEL_MAX_H = 330;
const PANEL_MAX_H = 520;

const is_browser_mode = () => window.innerWidth > 300 && !host.isNativeWindow;

if (!is_browser_mode()) {
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

  const in_browser = is_browser_mode();
  const [label_pinned, set_label_pinned] = createSignal(false);
  const [badge_hovered, set_badge_hovered] = createSignal(false);
  const [label_hovered, set_label_hovered] = createSignal(false);
  const label_visible = () => label_pinned() || badge_hovered() || label_hovered();
  const [panel_pinned, set_panel_pinned] = createSignal(false);
  const [dot_hovered, set_dot_hovered] = createSignal(false);
  const [panel_hovered, set_panel_hovered] = createSignal(false);
  const panel_visible = () => panel_pinned() || dot_hovered() || panel_hovered();

  const ws = ws_store_new();

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
  });

  createEffect(() => {
    const count = ws.label_grid_requested();
    if (count > 0) {
      if (in_browser) {
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
      if (in_browser) {
        set_label_pinned(true);
      } else {
        host.invoke({ cmd: "toggleLabelGrid" });
      }
    });
  });

  return (
    <div
      style={{
        ...(in_browser
          ? {
              display: "flex",
              "flex-direction": "column",
              "align-items": "flex-end",
              "padding-top": "16px",
              "padding-right": "16px",
              "min-height": "100vh",
              background: "url('/bliss.png') center/cover no-repeat fixed",
            }
          : {}),
      }}
    >
      <div
        style={{
          display: "flex",
          "flex-direction": "column",
          "align-items": "flex-end",
        }}
      >
        <div
          style={{
            background: "rgba(15, 17, 23, 0.5)",
            "backdrop-filter": "blur(20px)",
            "-webkit-backdrop-filter": "blur(20px)",
            width: in_browser ? `${COMPACT_W}px` : "100%",
            ...(in_browser
              ? { "box-shadow": "0 4px 24px rgba(0, 0, 0, 0.5)" }
              : { height: "100vh" }),
            overflow: "hidden",
            "border-radius": "20px",
          }}
        >
          <div
            class="pywebview-drag-region"
            style={{
              display: "flex",
              "align-items": "center",
              "justify-content": "center",
              padding: "0 8px 0 12px",
              height: "30px",
              "user-select": "none",
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
                in_browser
                  ? () => {
                      permission_ensure_once();
                      set_panel_pinned((v) => !v);
                    }
                  : () => {
                      permission_ensure_once();
                      host.invoke({ cmd: "toggleStatePanel" });
                    }
              }
              on_show_panel={
                in_browser
                  ? () => set_dot_hovered(true)
                  : () => {
                      host.invoke({ cmd: "showStatePanel" });
                    }
              }
              on_hide_panel={
                in_browser
                  ? () => set_dot_hovered(false)
                  : () => {
                      host.invoke({ cmd: "hideStatePanel" });
                    }
              }
              on_toggle_label={
                in_browser
                  ? () => {
                      permission_ensure_once();
                      set_label_pinned((v) => !v);
                    }
                  : () => {
                      permission_ensure_once();
                      host.invoke({ cmd: "toggleLabelGrid" });
                    }
              }
              on_show_label={
                in_browser
                  ? () => set_badge_hovered(true)
                  : () => {
                      host.invoke({ cmd: "showLabelGrid" });
                    }
              }
              on_hide_label={
                in_browser
                  ? () => set_badge_hovered(false)
                  : () => {
                      host.invoke({ cmd: "hideLabelGrid" });
                    }
              }
            />
          </div>
        </div>

        <Show when={in_browser && label_visible()}>
          {/* biome-ignore lint/a11y/noStaticElementInteractions: hover container */}
          <div
            onMouseEnter={() => set_label_hovered(true)}
            onMouseLeave={() => set_label_hovered(false)}
            style={{
              width: `${CONTENT_W}px`,
              "max-height": `${LABEL_MAX_H}px`,
              "overflow-y": "auto",
              "margin-top": "4px",
              background: "var(--bg)",
              "border-radius": "12px",
            }}
          >
            <LabelRecorder
              on_collapse={() => {
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
      </div>

      <Show when={in_browser && panel_visible()}>
        {/* biome-ignore lint/a11y/noStaticElementInteractions: hover container */}
        <div
          onMouseEnter={() => set_panel_hovered(true)}
          onMouseLeave={() => set_panel_hovered(false)}
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
              if (in_browser) {
                set_label_pinned(true);
                set_badge_hovered(true);
              } else {
                host.invoke({ cmd: "showLabelGrid" });
              }
            }}
          />
        </div>
      </Show>
    </div>
  );
};

export default App;
