import {
  app,
  BrowserWindow,
  ipcMain,
  Menu,
  Tray,
  nativeImage,
  screen,
  shell,
  dialog,
  Notification,
} from "electron";
import { spawn, type ChildProcess } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import {
  checkForUpdate,
  downloadAndApplyUpdate,
  getActivePayloadBackendPath,
  isPayloadDirPresent,
  lastManifestCheckFailure,
  type Manifest,
  type UpdateProgressEvent,
} from "./updater";

/** Must match `build.productName` in package.json (Finder / .app name, tray, notifications). */
function readAppDisplayName(): string {
  const pkgPath = path.join(__dirname, "..", "package.json");
  const raw = fs.readFileSync(pkgPath, "utf-8");
  const pkg = JSON.parse(raw) as { build?: { productName?: string } };
  return pkg.build?.productName ?? "taskclf";
}

const APP_DISPLAY_NAME = readAppDisplayName();

type HostCommand = {
  cmd: string;
  mode?: string;
  message?: string;
};

const COMPACT_SIZE = { width: 150, height: 30 };
const LABEL_SIZE = { width: 280, height: 330 };
const PANEL_SIZE = { width: 280, height: 520 };
const CHILD_GAP = 4;
const CHILD_HIDE_DELAY_MS = 300;
const DRAG_TOLERANCE = 10;
const WINDOW_MARGIN = 16;

let mainWindow: BrowserWindow | null = null;
let tray: Tray | null = null;
let sidecar: ChildProcess | null = null;
let isQuitting = false;
let hasShownWindow = false;

// ── Child window state machine ──────────────────────────────────────────

type ChildWindowState = {
  window: BrowserWindow | null;
  visible: boolean;
  pinned: boolean;
  hideTimer: ReturnType<typeof setTimeout> | null;
  expectedPos: { x: number; y: number } | null;
  size: { width: number; height: number };
  positionFn: () => void;
};

const labelChild: ChildWindowState = {
  window: null,
  visible: false,
  pinned: false,
  hideTimer: null,
  expectedPos: null,
  size: LABEL_SIZE,
  positionFn: () => {
    positionLabel();
  },
};

const panelChild: ChildWindowState = {
  window: null,
  visible: false,
  pinned: false,
  hideTimer: null,
  expectedPos: null,
  size: PANEL_SIZE,
  positionFn: () => {
    positionPanel();
  },
};

function childTimerCancel(child: ChildWindowState): void {
  if (child.hideTimer !== null) {
    clearTimeout(child.hideTimer);
    child.hideTimer = null;
  }
}

function childVisibilityOn(child: ChildWindowState): void {
  if (child.window === null || mainWindow === null) {
    return;
  }
  childTimerCancel(child);
  if (!child.visible) {
    child.visible = true;
    child.positionFn();
    child.window.showInactive();
  }
}

function childVisibilityOff(child: ChildWindowState): void {
  child.hideTimer = null;
  if (child.window !== null) {
    child.window.hide();
  }
  child.visible = false;
  child.pinned = false;
  child.expectedPos = null;
}

function childVisibilityOffDeferred(child: ChildWindowState): void {
  if (child.pinned) {
    return;
  }
  childTimerCancel(child);
  child.hideTimer = setTimeout(() => {
    childVisibilityOff(child);
  }, CHILD_HIDE_DELAY_MS);
}

function childPinToggle(child: ChildWindowState): void {
  if (child.window === null || mainWindow === null) {
    return;
  }
  if (child.visible && child.pinned) {
    child.pinned = false;
    childVisibilityOff(child);
  } else if (child.visible && !child.pinned) {
    child.pinned = true;
  } else {
    childTimerCancel(child);
    child.pinned = true;
    child.visible = true;
    child.positionFn();
    child.window.showInactive();
  }
}

function childDragDetected(child: ChildWindowState): boolean {
  if (!child.visible || child.window === null || child.expectedPos === null) {
    return false;
  }
  const bounds = child.window.getBounds();
  return (
    Math.abs(bounds.x - child.expectedPos.x) > DRAG_TOLERANCE
    || Math.abs(bounds.y - child.expectedPos.y) > DRAG_TOLERANCE
  );
}

// ── Child positioning ───────────────────────────────────────────────────

function positionLabel(): void {
  if (mainWindow === null || labelChild.window === null || !labelChild.visible) {
    return;
  }
  const pill = mainWindow.getBounds();
  const x = pill.x + COMPACT_SIZE.width - LABEL_SIZE.width;
  const y = pill.y + COMPACT_SIZE.height + CHILD_GAP;
  labelChild.window.setBounds({ x, y, ...LABEL_SIZE }, false);
  labelChild.expectedPos = { x, y };
}

function positionPanel(): void {
  if (mainWindow === null || panelChild.window === null || !panelChild.visible) {
    return;
  }
  const pill = mainWindow.getBounds();
  const rightX = pill.x + COMPACT_SIZE.width;
  let y = pill.y + COMPACT_SIZE.height + CHILD_GAP;
  if (labelChild.visible) {
    y += LABEL_SIZE.height + CHILD_GAP;
  }
  const x = rightX - PANEL_SIZE.width;
  panelChild.window.setBounds({ x, y, ...PANEL_SIZE }, false);
  panelChild.expectedPos = { x, y };
}

function onMainWindowMoved(): void {
  if (!childDragDetected(labelChild)) {
    positionLabel();
  }
  if (!childDragDetected(panelChild)) {
    positionPanel();
  }
}

// ── Environment helpers ─────────────────────────────────────────────────

function envString(name: string, fallback: string): string {
  const value = process.env[name];
  return value && value.length > 0 ? value : fallback;
}

function envInt(name: string, fallback: number): number {
  const value = process.env[name];
  if (!value) {
    return fallback;
  }
  const parsed = Number.parseInt(value, 10);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function envFlag(name: string): boolean {
  return process.env[name] === "1";
}

/** Set `TASKCLF_ELECTRON_DEBUG=1` for verbose console output (spawn line, waitForShell progress). */
function electronDebugEnabled(): boolean {
  return process.env.TASKCLF_ELECTRON_DEBUG === "1";
}

/** Persistent log path under userData (only valid after app ready). */
function launcherLogFilePath(): string {
  return path.join(app.getPath("userData"), "logs", "electron-launcher.log");
}

/** Append to console and `logs/electron-launcher.log` for post-mortem debugging. */
function launcherLog(message: string, level: "info" | "error" = "info"): void {
  const ts = new Date().toISOString();
  const line = `[${ts}] [${level}] ${message}`;
  if (level === "error") {
    console.error(line);
  } else {
    console.log(line);
  }
  try {
    const logDir = path.join(app.getPath("userData"), "logs");
    fs.mkdirSync(logDir, { recursive: true });
    fs.appendFileSync(path.join(logDir, "electron-launcher.log"), `${line}\n`, "utf8");
  } catch {
    // ignore disk errors
  }
}

/** Models dir for tray: absolute under userData when packaged (GUI apps often have cwd `/`). */
function defaultModelsDir(): string {
  const fromEnv = process.env.TASKCLF_ELECTRON_MODELS_DIR;
  if (fromEnv && fromEnv.length > 0) {
    return fromEnv;
  }
  if (app.isPackaged) {
    return path.join(app.getPath("userData"), "models");
  }
  return "models";
}

function sidecarExecutable(): string {
  if (app.isPackaged) {
    const activePath = getActivePayloadBackendPath();
    if (activePath) {
      return activePath;
    }
    throw new Error(
      `${APP_DISPLAY_NAME} core not found. An internet connection is required on first launch to download the application payload.`,
    );
  }
  return envString("TASKCLF_ELECTRON_PYTHON_EXECUTABLE", "python3");
}

function uiPort(): number {
  return envInt("TASKCLF_ELECTRON_UI_PORT", 8741);
}

function devMode(): boolean {
  return envFlag("TASKCLF_ELECTRON_DEV");
}

function shellUrl(): string {
  if (devMode()) {
    return "http://127.0.0.1:5173";
  }
  return `http://127.0.0.1:${uiPort()}`;
}

function sidecarArgs(): string[] {
  const args: string[] = [];
  if (!app.isPackaged) {
    args.push("-m", "taskclf.cli.main");
  }
  args.push(
    "tray",
    "--browser",
    "--no-tray",
    "--no-open-browser",
    "--port",
    String(uiPort()),
    "--aw-host",
    envString("TASKCLF_ELECTRON_AW_HOST", "http://localhost:5600"),
    "--poll-seconds",
    String(envInt("TASKCLF_ELECTRON_POLL_SECONDS", 60)),
    "--title-salt",
    envString("TASKCLF_ELECTRON_TITLE_SALT", "taskclf-default-salt"),
    "--transition-minutes",
    String(envInt("TASKCLF_ELECTRON_TRANSITION_MINUTES", 3)),
    "--models-dir",
    defaultModelsDir(),
  );

  const dataDir = process.env.TASKCLF_ELECTRON_DATA_DIR;
  if (dataDir) {
    args.push("--data-dir", dataDir);
  }

  const modelDir = process.env.TASKCLF_ELECTRON_MODEL_DIR;
  if (modelDir) {
    args.push("--model-dir", modelDir);
  }

  const username = process.env.TASKCLF_ELECTRON_USERNAME;
  if (username) {
    args.push("--username", username);
  }

  const retrainConfig = process.env.TASKCLF_ELECTRON_RETRAIN_CONFIG;
  if (retrainConfig) {
    args.push("--retrain-config", retrainConfig);
  }

  if (devMode()) {
    args.push("--dev");
  }

  return args;
}

// ── Window factories ────────────────────────────────────────────────────

function initialPillBounds(): Electron.Rectangle {
  const display = screen.getDisplayNearestPoint(screen.getCursorScreenPoint());
  const workArea = display.workArea;
  return {
    x: workArea.x + workArea.width - COMPACT_SIZE.width - WINDOW_MARGIN,
    y: workArea.y + WINDOW_MARGIN,
    ...COMPACT_SIZE,
  };
}

function getAppIconPath(): string {
  return path.join(__dirname, "..", "build", "icon.png");
}

function preloadScriptPath(): string {
  return path.join(__dirname, "preload.js");
}

function createPillWindow(): BrowserWindow {
  const preload = preloadScriptPath();
  const window = new BrowserWindow({
    ...initialPillBounds(),
    show: false,
    frame: false,
    transparent: true,
    alwaysOnTop: true,
    skipTaskbar: true,
    resizable: false,
    hasShadow: true,
    backgroundColor: "#00000000",
    icon: getAppIconPath(),
    webPreferences: {
      preload,
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: false,
    },
  });

  window.on("close", (event) => {
    if (isQuitting) {
      return;
    }
    event.preventDefault();
    window.hide();
  });

  window.on("closed", () => {
    mainWindow = null;
  });

  window.on("move", onMainWindowMoved);

  return window;
}

function createPopupWindow(size: { width: number; height: number }): BrowserWindow {
  const preload = preloadScriptPath();
  const window = new BrowserWindow({
    width: size.width,
    height: size.height,
    show: false,
    frame: false,
    transparent: true,
    alwaysOnTop: true,
    skipTaskbar: true,
    resizable: false,
    hasShadow: true,
    backgroundColor: "#00000000",
    icon: getAppIconPath(),
    webPreferences: {
      preload,
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: false,
    },
  });

  window.on("close", (event) => {
    if (isQuitting) {
      return;
    }
    event.preventDefault();
    window.hide();
  });

  return window;
}

// ── Show / toggle ───────────────────────────────────────────────────────

function showWindow(): void {
  if (mainWindow === null) {
    return;
  }
  if (!hasShownWindow) {
    mainWindow.setBounds(initialPillBounds(), false);
    hasShownWindow = true;
  }
  mainWindow.show();
  mainWindow.focus();
}

function toggleWindow(): void {
  if (mainWindow === null) {
    return;
  }
  if (mainWindow.isVisible()) {
    if (labelChild.visible) {
      childVisibilityOff(labelChild);
    }
    if (panelChild.visible) {
      childVisibilityOff(panelChild);
    }
    mainWindow.hide();
    return;
  }
  showWindow();
}

// ── Sidecar ─────────────────────────────────────────────────────────────

function spawnSidecar(): void {
  const exe = sidecarExecutable();
  const args = sidecarArgs();
  const childCwd = app.isPackaged ? app.getPath("userData") : process.cwd();
  const spawnSummary = `spawn exe=${exe} cwd=${childCwd} args=${JSON.stringify(args)}`;
  launcherLog(`[sidecar] ${spawnSummary}`, "info");
  if (app.isPackaged || electronDebugEnabled()) {
    console.log(`[sidecar] ${exe} ${args.join(" ")}`);
  }
  const env: NodeJS.ProcessEnv = {
    ...process.env,
    TASKCLF_ELECTRON_SHELL: "1",
  };
  if (app.isPackaged) {
    env.PYTHONUNBUFFERED = "1";
  }
  sidecar = spawn(exe, args, {
    stdio: "inherit",
    env,
    cwd: childCwd,
  });
  sidecar.once("error", (err) => {
    const msg = err instanceof Error ? err.message : String(err);
    launcherLog(`tray backend spawn error: ${msg}`, "error");
    console.error(`${APP_DISPLAY_NAME} tray backend spawn error:`, err);
    console.error(`(See ${launcherLogFilePath()} for history.)`);
  });
  sidecar.once("exit", (code, signal) => {
    sidecar = null;
    if (isQuitting) {
      return;
    }
    const detail = `exitCode=${code ?? "null"} signal=${signal ?? "null"} exe=${exe}`;
    launcherLog(`tray backend exited unexpectedly: ${detail}`, "error");
    launcherLog(`hint: set TASKCLF_ELECTRON_DEBUG=1 and re-run from Terminal; log file: ${launcherLogFilePath()}`, "error");
    const sigPart = signal ? ` signal=${signal}` : "";
    console.error(
      `${APP_DISPLAY_NAME} tray backend exited unexpectedly (code=${code ?? "unknown"}${sigPart})`,
    );
    console.error(`Full detail: ${detail}`);
    console.error(`Log file: ${launcherLogFilePath()}`);
    app.quit();
  });
}

async function stopSidecar(): Promise<void> {
  if (sidecar === null || sidecar.killed) {
    return;
  }
  const proc = sidecar;

  await new Promise<void>((resolve) => {
    let settled = false;
    const finish = () => {
      if (settled) {
        return;
      }
      settled = true;
      resolve();
    };

    proc.once("exit", finish);
    proc.kill("SIGTERM");

    setTimeout(() => {
      if (sidecar !== null && !sidecar.killed) {
        sidecar.kill("SIGKILL");
      }
      finish();
    }, 5000);
  });
}

async function waitForShell(url: string, timeoutMs = 30000): Promise<void> {
  const deadline = Date.now() + timeoutMs;
  let attempts = 0;
  let lastDetail = "unknown";
  while (Date.now() < deadline) {
    attempts++;
    try {
      const response = await fetch(url, { method: "GET" });
      if (response.ok) {
        if (electronDebugEnabled()) {
          launcherLog(`waitForShell: OK ${url} after ${attempts} attempt(s)`, "info");
        }
        return;
      }
      lastDetail = `HTTP ${response.status} ${response.statusText}`;
    } catch (e) {
      lastDetail = e instanceof Error ? e.message : String(e);
    }
    if (electronDebugEnabled() && attempts % 10 === 1) {
      launcherLog(`waitForShell: still waiting for ${url} (attempt ${attempts}, last: ${lastDetail})`, "info");
    }
    await new Promise((resolve) => setTimeout(resolve, 500));
  }
  const errMsg = `Timed out waiting for ${url} after ${attempts} attempts (last: ${lastDetail})`;
  launcherLog(errMsg, "error");
  throw new Error(errMsg);
}

async function sidecarRequest(
  pathName: string,
  init?: RequestInit,
): Promise<Response | null> {
  try {
    return await fetch(`http://127.0.0.1:${uiPort()}${pathName}`, init);
  } catch {
    return null;
  }
}

async function importLabels(): Promise<void> {
  const result = await dialog.showOpenDialog({
    title: "Import Labels",
    filters: [{ name: "CSV files", extensions: ["csv"] }, { name: "All files", extensions: ["*"] }],
    properties: ["openFile"],
  });
  if (result.canceled || result.filePaths.length === 0) return;
  const filePath = result.filePaths[0];

  const strategyResult = await dialog.showMessageBox({
    type: "question",
    title: "Import Strategy",
    message: "Merge with existing labels?\n\nYes = merge (keep existing, add new)\nNo = overwrite (replace all labels)",
    buttons: ["Cancel", "Overwrite", "Merge"],
    defaultId: 2,
    cancelId: 0,
  });
  if (strategyResult.response === 0) return;
  const strategy = strategyResult.response === 2 ? "merge" : "overwrite";

  try {
    const buffer = fs.readFileSync(filePath);
    const blob = new Blob([buffer], { type: "text/csv" });
    const formData = new FormData();
    formData.append("file", blob, path.basename(filePath));
    formData.append("strategy", strategy);

    const res = await fetch(`http://127.0.0.1:${uiPort()}/api/labels/import`, {
      method: "POST",
      body: formData,
    });
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();
    new Notification({ title: APP_DISPLAY_NAME, body: `Imported ${data.imported} labels` }).show();
  } catch (err) {
    new Notification({ title: APP_DISPLAY_NAME, body: `Import failed: ${err}` }).show();
  }
}

async function exportLabels(): Promise<void> {
  const result = await dialog.showSaveDialog({
    title: "Export Labels",
    defaultPath: "labels_export.csv",
    filters: [{ name: "CSV files", extensions: ["csv"] }, { name: "All files", extensions: ["*"] }],
  });
  if (result.canceled || !result.filePath) return;
  const filePath = result.filePath;

  try {
    const res = await fetch(`http://127.0.0.1:${uiPort()}/api/labels/export`);
    if (!res.ok) throw new Error(await res.text());
    const buffer = await res.arrayBuffer();
    fs.writeFileSync(filePath, Buffer.from(buffer));
    new Notification({ title: APP_DISPLAY_NAME, body: `Labels exported to ${path.basename(filePath)}` }).show();
  } catch (err) {
    new Notification({ title: APP_DISPLAY_NAME, body: `Export failed: ${err}` }).show();
  }
}

let syncTimer: ReturnType<typeof setInterval> | null = null;
let updateCheckTimer: ReturnType<typeof setInterval> | null = null;
let updatePromptShownForVersion: string | null = null;

async function performBackgroundUpdateCheck() {
  if (!app.isPackaged || isQuitting || mainWindow === null) {
    return;
  }

  try {
    const manifest = await checkForUpdate();
    if (manifest && manifest.version !== updatePromptShownForVersion) {
      updatePromptShownForVersion = manifest.version;

      const res = await dialog.showMessageBox({
        type: "info",
        title: "Update Available",
        message: `A new version of ${APP_DISPLAY_NAME} (v${manifest.version}) is available.`,
        detail: "Would you like to download and restart now?",
        buttons: ["Update and Restart", "Later"],
        defaultId: 0,
        cancelId: 1
      });

      if (res.response === 0) {
        try {
          if (isPayloadDirPresent(manifest)) {
            await downloadAndApplyUpdate(manifest);
          } else {
            await runDownloadWithProgress(`Downloading ${APP_DISPLAY_NAME} update`, manifest);
          }
          app.relaunch();
          app.quit();
        } catch (err) {
          const msg = err instanceof Error ? err.message : String(err);
          launcherLog(`background update apply failed: ${msg}`, "error");
          dialog.showErrorBox(
            "Update Failed",
            `Failed to apply update: ${msg}\n\nLog file:\n${launcherLogFilePath()}`,
          );
        }
      }
    }
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    launcherLog(`background update check failed: ${msg}`, "error");
    console.error("[main] Background update check failed:", err);
  }
}

async function syncTrayMenu() {
  if (!tray || isQuitting) return;
  try {
    const stateRes = await fetch(`http://127.0.0.1:${uiPort()}/api/tray/state`);
    if (!stateRes.ok) return;
    const state = await stateRes.json();

    const modelsRes = await fetch(`http://127.0.0.1:${uiPort()}/api/train/models`);
    let models: any[] = [];
    if (modelsRes.ok) {
      models = await modelsRes.json();
    }

    const template: Electron.MenuItemConstructorOptions[] = [
      { label: "Toggle Dashboard", click: toggleWindow },
      { label: state.paused ? "Resume" : "Pause", click: () => sidecarRequest("/api/tray/action/pause_toggle", { method: "POST" }) },
      { type: "separator" },
      { label: "Label Stats", click: () => sidecarRequest("/api/tray/action/label_stats", { method: "POST" }) },
      { label: "Import Labels", click: importLabels },
      { label: "Export Labels", click: exportLabels },
      { type: "separator" },
      {
        label: "Model",
        submenu: [
          ...models.filter((m: any) => m.valid).map((m: any) => ({
            label: m.model_id,
            type: "checkbox" as const,
            checked: state.model_dir === m.path,
            click: () => sidecarRequest("/api/tray/action/switch_model", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ model_id: m.model_id })
            })
          })),
          ...(models.filter((m: any) => m.valid).length > 0 ? [
            {
              label: "(No Model)",
              type: "checkbox" as const,
              checked: state.model_dir === null,
              click: () => sidecarRequest("/api/tray/action/unload_model", { method: "POST" })
            }
          ] : [
            { label: "(no models found)", enabled: false }
          ]),
          { type: "separator" },
          { label: "Reload Model", enabled: state.model_dir !== null, click: () => sidecarRequest("/api/tray/action/reload_model", { method: "POST" }) },
          { label: "Check Retrain", click: () => sidecarRequest("/api/tray/action/check_retrain", { method: "POST" }) }
        ]
      },
      { label: "Status", click: () => sidecarRequest("/api/tray/action/show_status", { method: "POST" }) },
      { label: "Open Data Folder", click: () => sidecarRequest("/api/tray/action/open_data_dir", { method: "POST" }) },
      { label: "Edit Config", click: () => sidecarRequest("/api/tray/action/edit_config", { method: "POST" }) },
      { label: "Report Issue", click: () => sidecarRequest("/api/tray/action/report_issue", { method: "POST" }) },
      { type: "separator" },
      { label: "Quit", click: () => { isQuitting = true; void stopSidecar().finally(() => app.quit()); } }
    ];

    tray.setContextMenu(Menu.buildFromTemplate(template));
  } catch (err) {
    // Ignore errors, server might not be ready
  }
}

// ── Tray ────────────────────────────────────────────────────────────────

function createTrayIcon(): Electron.NativeImage {
  const svg = `
    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 16 16">
      <circle cx="8" cy="8" r="6" fill="#4caf50" />
      <rect x="7" y="3" width="2" height="6" rx="1" fill="#0f1117" />
      <circle cx="8" cy="11" r="1.2" fill="#0f1117" />
    </svg>
  `.trim();
  return nativeImage.createFromDataURL(
    `data:image/svg+xml;charset=utf-8,${encodeURIComponent(svg)}`,
  );
}

function createTray(): Tray {
  const instance = new Tray(createTrayIcon());
  instance.setToolTip(APP_DISPLAY_NAME);
  instance.on("click", () => {
    toggleWindow();
  });
  instance.setContextMenu(
    Menu.buildFromTemplate([
      {
        label: "Toggle Dashboard",
        click: () => {
          toggleWindow();
        },
      },
      { type: "separator" },
      {
        label: "Quit",
        click: () => {
          isQuitting = true;
          void stopSidecar().finally(() => app.quit());
        },
      },
    ]),
  );
  return instance;
}

function errorPageHtml(message: string): string {
  const escaped = message
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
  return `
    <html>
      <body style="margin:0;background:#101218;color:#f5f5f5;font-family:-apple-system,BlinkMacSystemFont,sans-serif;">
        <div style="padding:20px">
          <h3 style="margin:0 0 12px">${APP_DISPLAY_NAME} (Electron)</h3>
          <p style="margin:0;line-height:1.5">${escaped}</p>
        </div>
      </body>
    </html>
  `;
}

function escapeHtmlAttr(text: string): string {
  return text
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function formatBytes(n: number): string {
  if (n < 1024) {
    return `${n} B`;
  }
  if (n < 1024 * 1024) {
    return `${(n / 1024).toFixed(1)} KB`;
  }
  return `${(n / (1024 * 1024)).toFixed(1)} MB`;
}

function formatProgressLine(event: UpdateProgressEvent): string {
  switch (event.phase) {
    case "download": {
      const r = event.receivedBytes ?? 0;
      const t = event.totalBytes;
      const pct = event.percent;
      const parts: string[] = [];
      if (pct != null) {
        parts.push(`${pct}%`);
      }
      if (t != null && t > 0) {
        parts.push(`${formatBytes(r)} / ${formatBytes(t)}`);
      } else {
        parts.push(`${formatBytes(r)} received`);
      }
      return parts.join(" · ");
    }
    case "verify":
      return "Verifying…";
    case "extract":
      return "Extracting…";
    default:
      return "";
  }
}

function downloadProgressPageHtml(heading: string): string {
  const h = escapeHtmlAttr(heading);
  return `<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  body { margin:0; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; padding:16px 20px; background:#f5f5f5; color:#111; }
  h1 { font-size:15px; font-weight:600; margin:0 0 12px; }
  #barwrap { height:10px; background:#ddd; border-radius:5px; overflow:hidden; position:relative; }
  #barwrap.indeterminate #bar { width:35% !important; animation:slide 1.2s ease-in-out infinite; }
  @keyframes slide { 0% { transform:translateX(-100%); } 100% { transform:translateX(400%); } }
  #bar { height:100%; width:0%; background:#2563eb; border-radius:5px; transition:width 0.12s ease; }
  #line2 { margin:10px 0 0; font-size:13px; color:#444; min-height:1.2em; }
</style>
</head>
<body>
  <h1>${h}</h1>
  <div id="barwrap"><div id="bar"></div></div>
  <p id="line2"></p>
</body>
</html>`;
}

function createDownloadProgressWindow(heading: string): BrowserWindow {
  const win = new BrowserWindow({
    width: 440,
    height: 170,
    show: false,
    center: true,
    resizable: false,
    minimizable: false,
    maximizable: false,
    title: heading,
    icon: getAppIconPath(),
    webPreferences: {
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
    },
  });
  win.loadURL(
    `data:text/html;charset=utf-8,${encodeURIComponent(downloadProgressPageHtml(heading))}`,
  );
  win.once("ready-to-show", () => {
    win.show();
  });
  return win;
}

async function applyProgressToWindow(win: BrowserWindow, event: UpdateProgressEvent): Promise<void> {
  if (win.isDestroyed()) {
    return;
  }
  const pct = event.percent;
  const line2 = formatProgressLine(event);
  const js = `(function(){
    var p = ${pct === null || pct === undefined ? "null" : String(Math.round(pct))};
    var line2 = ${JSON.stringify(line2)};
    var bar = document.getElementById("bar");
    var wrap = document.getElementById("barwrap");
    if (bar && wrap) {
      if (p === null) {
        wrap.className = "indeterminate";
        bar.style.width = "35%";
      } else {
        wrap.className = "";
        bar.style.width = p + "%";
      }
    }
    var el = document.getElementById("line2");
    if (el) { el.textContent = line2; }
  })();`;
  await win.webContents.executeJavaScript(js, true);
}

async function runDownloadWithProgress(heading: string, manifest: Manifest): Promise<void> {
  const win = createDownloadProgressWindow(heading);
  try {
    await downloadAndApplyUpdate(manifest, {
      onProgress: (e) => applyProgressToWindow(win, e),
    });
  } finally {
    if (!win.isDestroyed()) {
      win.close();
    }
  }
}

// ── IPC ─────────────────────────────────────────────────────────────────

ipcMain.handle("taskclf-host", async (_event, command: HostCommand) => {
  switch (command.cmd) {
    case "toggleDashboard":
      toggleWindow();
      return;
    case "hideWindow":
      mainWindow?.hide();
      return;
    case "showLabelGrid":
      childVisibilityOn(labelChild);
      return;
    case "hideLabelGrid":
      childVisibilityOffDeferred(labelChild);
      return;
    case "toggleLabelGrid":
      childPinToggle(labelChild);
      return;
    case "cancelLabelHide":
      childTimerCancel(labelChild);
      return;
    case "showStatePanel":
      childVisibilityOn(panelChild);
      return;
    case "hideStatePanel":
      childVisibilityOffDeferred(panelChild);
      return;
    case "toggleStatePanel":
      childPinToggle(panelChild);
      return;
    case "cancelPanelHide":
      childTimerCancel(panelChild);
      return;
    case "setWindowMode":
      return;
    case "frontendDebugLog":
      console.debug("[frontend]", command.message ?? "");
      return;
    case "frontendErrorLog":
      console.error("[frontend]", command.message ?? "");
      return;
    default:
      return;
  }
});

// ── Bootstrap ───────────────────────────────────────────────────────────

async function start(): Promise<void> {
  launcherLog(
    `start: packaged=${app.isPackaged} version=${app.getVersion()} userData=${app.getPath("userData")}`,
    "info",
  );

  app.setName(APP_DISPLAY_NAME);
  if (process.platform === "darwin") {
    app.dock?.setIcon(getAppIconPath());
    app.dock?.hide();
  }

  tray = createTray();
  mainWindow = createPillWindow();

  const labelWin = createPopupWindow(LABEL_SIZE);
  labelChild.window = labelWin;
  labelWin.on("closed", () => {
    labelChild.window = null;
  });

  const panelWin = createPopupWindow(PANEL_SIZE);
  panelChild.window = panelWin;
  panelWin.on("closed", () => {
    panelChild.window = null;
  });

  if (app.isPackaged) {
    const isFirstRun = getActivePayloadBackendPath() === null;

    if (isFirstRun) {
      const manifest = await checkForUpdate();
      if (!manifest) {
        const logHint = `\n\nLog file:\n${launcherLogFilePath()}`;
        const reason = lastManifestCheckFailure
          ? `\n\nDetails:\n${lastManifestCheckFailure}`
          : "\n\n(No manifest was returned. Check that the latest GitHub release includes manifest.json.)";
        launcherLog(`first run: manifest unavailable${reason}`, "error");
        dialog.showErrorBox(
          "Network Error",
          `${APP_DISPLAY_NAME} core not found. An internet connection is required on first launch to download the application payload.${reason}${logHint}`,
        );
        app.quit();
        return;
      }

      const res = await dialog.showMessageBox({
        type: "info",
        title: "Initial Setup",
        message: `${APP_DISPLAY_NAME} needs to download its core components before starting.`,
        detail: `Version: ${manifest.version}`,
        buttons: ["Download and Start", "Quit"],
        defaultId: 0,
        cancelId: 1
      });

      if (res.response !== 0) {
        app.quit();
        return;
      }

      try {
        if (isPayloadDirPresent(manifest)) {
          await downloadAndApplyUpdate(manifest);
        } else {
          await runDownloadWithProgress(`Downloading ${APP_DISPLAY_NAME} core`, manifest);
        }
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        launcherLog(`first run: download/apply failed: ${msg}`, "error");
        dialog.showErrorBox(
          "Update Failed",
          `Failed to download ${APP_DISPLAY_NAME} core: ${msg}\n\nLog file:\n${launcherLogFilePath()}`,
        );
        app.quit();
        return;
      }
    } else {
      // Check for updates in the background on startup
      performBackgroundUpdateCheck();
    }
  }

  try {
    spawnSidecar();

    const base = shellUrl();
    await waitForShell(base);
    await Promise.all([
      mainWindow.loadURL(base),
      labelWin.loadURL(`${base}?view=label`),
      panelWin.loadURL(`${base}?view=panel`),
    ]);
    showWindow();
    syncTimer = setInterval(() => { void syncTrayMenu(); }, 5000);
    updateCheckTimer = setInterval(() => { void performBackgroundUpdateCheck(); }, 60 * 60 * 1000); // 1 hour
    void syncTrayMenu();
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    launcherLog(`start failed: ${message}`, "error");
    await mainWindow.loadURL(
      `data:text/html;charset=utf-8,${encodeURIComponent(
        errorPageHtml(`${message}\n\nLog: ${launcherLogFilePath()}`),
      )}`,
    );
    showWindow();
  }
}

app.on("window-all-closed", () => {
  // Keep the tray app alive after the last window is hidden.
});

app.on("before-quit", () => {
  isQuitting = true;
  if (syncTimer !== null) {
    clearInterval(syncTimer);
    syncTimer = null;
  }
  if (updateCheckTimer !== null) {
    clearInterval(updateCheckTimer);
    updateCheckTimer = null;
  }
});

app.on("activate", () => {
  showWindow();
});

void app.whenReady().then(start);
