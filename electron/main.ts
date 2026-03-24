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

type WindowMode = "compact" | "label" | "panel" | "dashboard";

const COMPACT_SIZE = { width: 150, height: 30 };
const LABEL_WINDOW_HEIGHT = 30 + 4 + 330;
const PANEL_WINDOW_HEIGHT = 30 + 4 + 520;
const DASHBOARD_WINDOW_HEIGHT = 30 + 4 + 330 + 4 + 520;
const CONTENT_WIDTH = 280;
const WINDOW_MARGIN = 16;

let mainWindow: BrowserWindow | null = null;
let tray: Tray | null = null;
let sidecar: ChildProcess | null = null;
let isQuitting = false;
let windowMode: WindowMode = "compact";
let hasShownWindow = false;

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

function pythonExecutable(): string {
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
  const args = [
    "-m",
    "taskclf.cli.main",
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
  ];

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

function sizeForMode(mode: WindowMode): { width: number; height: number } {
  switch (mode) {
    case "label":
      return { width: CONTENT_WIDTH, height: LABEL_WINDOW_HEIGHT };
    case "panel":
      return { width: CONTENT_WIDTH, height: PANEL_WINDOW_HEIGHT };
    case "dashboard":
      return { width: CONTENT_WIDTH, height: DASHBOARD_WINDOW_HEIGHT };
    case "compact":
    default:
      return COMPACT_SIZE;
  }
}

function clamp(value: number, min: number, max: number): number {
  if (max < min) {
    return min;
  }
  return Math.min(Math.max(value, min), max);
}

function currentDisplayWorkArea(bounds: Electron.Rectangle): Electron.Rectangle {
  return screen.getDisplayMatching(bounds).workArea;
}

function initialBoundsFor(mode: WindowMode): Electron.Rectangle {
  const display = screen.getDisplayNearestPoint(screen.getCursorScreenPoint());
  const workArea = display.workArea;
  const size = sizeForMode(mode);
  return {
    x: workArea.x + workArea.width - size.width - WINDOW_MARGIN,
    y: workArea.y + WINDOW_MARGIN,
    width: size.width,
    height: size.height,
  };
}

function applyWindowMode(mode: WindowMode): void {
  windowMode = mode;
  if (mainWindow === null) {
    return;
  }

  const currentBounds = mainWindow.getBounds();
  const workArea = currentDisplayWorkArea(currentBounds);
  const nextSize = sizeForMode(mode);
  const rightEdge = currentBounds.x + currentBounds.width;
  const nextX = clamp(
    rightEdge - nextSize.width,
    workArea.x,
    workArea.x + workArea.width - nextSize.width,
  );
  const nextY = clamp(
    currentBounds.y,
    workArea.y,
    workArea.y + workArea.height - nextSize.height,
  );

  mainWindow.setBounds(
    {
      x: nextX,
      y: nextY,
      width: nextSize.width,
      height: nextSize.height,
    },
    false,
  );
}

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

function createWindow(): BrowserWindow {
  const preloadPath = path.join(__dirname, "preload.js");
  const window = new BrowserWindow({
    ...initialBoundsFor(windowMode),
    show: false,
    frame: false,
    transparent: true,
    alwaysOnTop: true,
    skipTaskbar: true,
    resizable: false,
    hasShadow: true,
    backgroundColor: "#00000000",
    webPreferences: {
      preload: preloadPath,
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

  return window;
}

function showWindow(): void {
  if (mainWindow === null) {
    return;
  }
  if (!hasShownWindow) {
    mainWindow.setBounds(initialBoundsFor(windowMode), false);
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
    mainWindow.hide();
    return;
  }
  showWindow();
}

function spawnSidecar(): void {
  sidecar = spawn(pythonExecutable(), sidecarArgs(), {
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

ipcMain.handle("taskclf-host", async (_event, command: HostCommand) => {
  switch (command.cmd) {
    case "toggleDashboard":
      toggleWindow();
      return;
    case "hideWindow":
      mainWindow?.hide();
      return;
    case "setWindowMode":
      if (
        command.mode === "compact"
        || command.mode === "label"
        || command.mode === "panel"
        || command.mode === "dashboard"
      ) {
        applyWindowMode(command.mode);
      }
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

async function start(): Promise<void> {
  app.setName("taskclf");
  if (process.platform === "darwin") {
    app.dock?.hide();
  }

  tray = createTray();
  mainWindow = createWindow();
  spawnSidecar();

  try {
    await waitForShell(shellUrl());
    await mainWindow.loadURL(shellUrl());
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
