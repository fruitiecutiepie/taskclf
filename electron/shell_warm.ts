import type { BrowserWindow } from "electron";

/** Inline placeholder while the FastAPI sidecar boots; warms the pill renderer in parallel with `waitForShell`. */
const SHELL_LOADING_HTML = `<!DOCTYPE html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>taskclf</title><style>body{font-family:system-ui,sans-serif;display:flex;align-items:center;justify-content:center;height:100vh;margin:0;background:#0f0f12;color:#e8e8ec;font-size:14px}@keyframes p{to{opacity:.35}}.s{animation:p 1s ease-in-out infinite alternate}</style></head><body><span class="s">Loading…</span></body></html>`;

export function shellLoadingDataUrl(): string {
  return `data:text/html;charset=utf-8,${encodeURIComponent(SHELL_LOADING_HTML)}`;
}

/** Load the placeholder into the pill window only (label/panel popups are created lazily on first use). */
export async function warmPillWindow(pill: BrowserWindow): Promise<void> {
  await pill.loadURL(shellLoadingDataUrl());
}
