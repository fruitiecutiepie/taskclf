import {
  app,
  BrowserWindow,
  ipcMain,
  Menu,
  Tray,
  nativeImage,
  screen,
  shell,
} from "electron";
import { spawn, type ChildProcess } from "node:child_process";
import path from "node:path";

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

function bundledBackendPath(): string {
  const name = process.platform === "win32" ? "entry.exe" : "entry";
  return path.join(process.resourcesPath, "backend", name);
}

function sidecarExecutable(): string {
  if (app.isPackaged) {
    return bundledBackendPath();
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
    envString("TASKCLF_ELECTRON_MODELS_DIR", "models"),
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
  sidecar = spawn(sidecarExecutable(), sidecarArgs(), {
    stdio: "inherit",
    env: process.env,
  });
  sidecar.once("exit", (code) => {
    sidecar = null;
    if (isQuitting) {
      return;
    }
    console.error(`taskclf tray backend exited unexpectedly (${code ?? "unknown"})`);
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
  while (Date.now() < deadline) {
    try {
      const response = await fetch(url, { method: "GET" });
      if (response.ok) {
        return;
      }
    } catch {
      // Server not ready yet.
    }
    await new Promise((resolve) => setTimeout(resolve, 500));
  }
  throw new Error(`Timed out waiting for ${url}`);
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
  instance.setToolTip("taskclf");
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
      {
        label: "Open In Browser",
        click: () => {
          void shell.openExternal(shellUrl());
        },
      },
      {
        label: "Toggle Pause",
        click: () => {
          void sidecarRequest("/api/tray/pause", { method: "POST" });
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
          <h3 style="margin:0 0 12px">taskclf Electron shell</h3>
          <p style="margin:0;line-height:1.5">${escaped}</p>
        </div>
      </body>
    </html>
  `;
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
  app.setName("taskclf");
  if (process.platform === "darwin") {
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

  spawnSidecar();

  try {
    const base = shellUrl();
    await waitForShell(base);
    await Promise.all([
      mainWindow.loadURL(base),
      labelWin.loadURL(`${base}?view=label`),
      panelWin.loadURL(`${base}?view=panel`),
    ]);
    showWindow();
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    await mainWindow.loadURL(
      `data:text/html;charset=utf-8,${encodeURIComponent(errorPageHtml(message))}`,
    );
    showWindow();
  }
}

app.on("window-all-closed", () => {
  // Keep the tray app alive after the last window is hidden.
});

app.on("before-quit", () => {
  isQuitting = true;
});

app.on("activate", () => {
  showWindow();
});

void app.whenReady().then(start);
