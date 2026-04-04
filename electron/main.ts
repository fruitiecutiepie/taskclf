import {
  app,
  BrowserWindow,
  clipboard,
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
  clearSelectedVersion,
  getActiveVersion,
  getActivePayloadBackendPath,
  getSelectedVersion,
  downloadAndApplyUpdate,
  isPayloadDirPresent,
  lastManifestCheckFailure,
  listInstalledPayloadVersions,
  resolvePayloadRelease,
  setSelectedVersion,
  useInstalledPayloadVersion,
  type Manifest,
  type PayloadResolution,
  type UpdateProgressEvent,
} from "./updater";
import { compareVersions } from "./update_policy";
import {
  installableOnlyPayloadVersions,
  orderedCompatiblePayloadVersions,
  payloadChooserOffersMultipleVersions,
} from "./payload_choice";

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
const DEFAULT_MANIFEST_TIMEOUT_MS = 15000;
const REPORT_ISSUE_URL_BASE = "https://github.com/fruitiecutiepie/taskclf/issues/new";
const MAX_REPORT_ISSUE_URL_LEN = 8000;
const MAX_RECENT_SIDECAR_LINES = 20;

let mainWindow: BrowserWindow | null = null;
let tray: Tray | null = null;
let sidecar: ChildProcess | null = null;
let isQuitting = false;
let hasShownWindow = false;
let fatalLaunchErrorShown = false;
const recentSidecarLines: string[] = [];
let latestPayloadResolution: PayloadResolution | null = null;

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

function manifestFetchTimeoutMs(): number {
  return Math.max(0, envInt("TASKCLF_MANIFEST_TIMEOUT_MS", DEFAULT_MANIFEST_TIMEOUT_MS));
}

/** Persistent log path under userData (only valid after app ready). */
function launcherLogFilePath(): string {
  return path.join(app.getPath("userData"), "logs", "electron-launcher.log");
}

/** Append to console and `logs/electron-launcher.log` for post-mortem debugging. */
function launcherLog(
  message: string,
  level: "info" | "error" = "info",
  options?: { echoToConsole?: boolean },
): void {
  const ts = new Date().toISOString();
  const line = `[${ts}] [${level}] ${message}`;
  if (options?.echoToConsole !== false) {
    if (level === "error") {
      console.error(line);
    } else {
      console.log(line);
    }
  }
  try {
    const logDir = path.join(app.getPath("userData"), "logs");
    fs.mkdirSync(logDir, { recursive: true });
    fs.appendFileSync(path.join(logDir, "electron-launcher.log"), `${line}\n`, "utf8");
  } catch {
    // ignore disk errors
  }
}

function rememberSidecarLine(line: string): void {
  if (line.length === 0) {
    return;
  }
  recentSidecarLines.push(line);
  if (recentSidecarLines.length > MAX_RECENT_SIDECAR_LINES) {
    recentSidecarLines.splice(0, recentSidecarLines.length - MAX_RECENT_SIDECAR_LINES);
  }
}

function readLauncherLogTail(maxLines = 30): string {
  try {
    const lines = fs.readFileSync(launcherLogFilePath(), "utf8").trimEnd().split(/\r?\n/);
    if (lines.length <= maxLines) {
      return lines.join("\n");
    }
    return lines.slice(-maxLines).join("\n");
  } catch {
    return "";
  }
}

function buildLauncherIssueUrl(title: string, detail: string): string {
  const params = new URLSearchParams({
    template: "bug_report.yml",
    title: `[Bug]: ${title}`,
    diagnostics: [
      `Launcher version: ${app.getVersion()}`,
      `Electron: ${process.versions.electron}`,
      `Platform: ${process.platform} ${process.arch}`,
      `Active payload: ${getActiveVersion() ?? "none"}`,
      `User data: ${app.getPath("userData")}`,
      `Launcher log: ${launcherLogFilePath()}`,
      "",
      detail,
    ].join("\n"),
  });

  const logTail = readLauncherLogTail();
  if (logTail) {
    params.set("logs", logTail);
  }

  let url = `${REPORT_ISSUE_URL_BASE}?${params.toString()}`;
  if (url.length > MAX_REPORT_ISSUE_URL_LEN) {
    params.delete("logs");
    url = `${REPORT_ISSUE_URL_BASE}?${params.toString()}`;
  }
  return url;
}

function fatalDialogDetail(message: string, detail?: string): string {
  const parts: string[] = [message];
  if (detail) {
    parts.push(`Details:\n${detail}`);
  }
  if (recentSidecarLines.length > 0) {
    parts.push(`Recent backend output:\n${recentSidecarLines.join("\n")}`);
  }
  parts.push(`Launcher log:\n${launcherLogFilePath()}`);
  return parts.join("\n\n");
}

async function showFatalLaunchError(
  title: string,
  message: string,
  detail?: string,
): Promise<void> {
  if (fatalLaunchErrorShown || isQuitting) {
    return;
  }
  fatalLaunchErrorShown = true;

  const combinedDetail = fatalDialogDetail(message, detail);
  const response = await dialog.showMessageBox({
    type: "error",
    title,
    message,
    detail: combinedDetail,
    buttons: ["Report Issue", "Open Log Folder", "Copy Details", "Quit"],
    defaultId: 0,
    cancelId: 3,
    noLink: true,
  });

  if (response.response === 0) {
    await shell.openExternal(buildLauncherIssueUrl(title, combinedDetail));
  } else if (response.response === 1) {
    shell.showItemInFolder(launcherLogFilePath());
  } else if (response.response === 2) {
    clipboard.writeText(combinedDetail);
  }

  isQuitting = true;
  void stopSidecar().finally(() => app.quit());
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

function uniquePayloadVersions(versions: Array<string | null>): string[] {
  const seen = new Set<string>();
  const result: string[] = [];
  for (const version of versions) {
    if (!version || seen.has(version)) {
      continue;
    }
    seen.add(version);
    result.push(version);
  }
  return result;
}

function activateBestLocalPayloadFallback(): string | null {
  const candidates = uniquePayloadVersions([
    getSelectedVersion(),
    getActiveVersion(),
    ...listInstalledPayloadVersions(),
  ]);
  for (const version of candidates) {
    if (useInstalledPayloadVersion(version)) {
      return version;
    }
  }
  return null;
}

function sidecarExecutable(): string {
  if (app.isPackaged) {
    const activePayloadPath = getActivePayloadBackendPath();
    if (activePayloadPath) {
      return activePayloadPath;
    }
    throw new Error(
      `${APP_DISPLAY_NAME} could not resolve a local payload to start.`,
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

function attachSidecarOutput(
  stream: NodeJS.ReadableStream | null,
  name: "stdout" | "stderr",
): void {
  if (stream === null) {
    return;
  }
  let pending = "";
  const level = name === "stderr" ? "error" : "info";
  const mirror = name === "stderr" ? process.stderr : process.stdout;

  const flush = (chunk: string): void => {
    if (chunk.length === 0) {
      return;
    }
    const combined = `${pending}${chunk}`;
    const parts = combined.split(/\r?\n/);
    pending = parts.pop() ?? "";
    for (const line of parts) {
      const trimmed = line.trimEnd();
      if (trimmed.length === 0) {
        continue;
      }
      rememberSidecarLine(`[${name}] ${trimmed}`);
      launcherLog(`[sidecar:${name}] ${trimmed}`, level, { echoToConsole: false });
    }
  };

  stream.on("data", (chunk) => {
    const text = Buffer.isBuffer(chunk) ? chunk.toString("utf8") : String(chunk);
    flush(text);
    if (mirror.writable) {
      mirror.write(text);
    }
  });
  stream.on("end", () => {
    if (pending.length === 0) {
      return;
    }
    rememberSidecarLine(`[${name}] ${pending}`);
    launcherLog(`[sidecar:${name}] ${pending}`, level, { echoToConsole: false });
    pending = "";
  });
}

function spawnSidecar(): void {
  recentSidecarLines.length = 0;
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
    stdio: app.isPackaged ? ["ignore", "pipe", "pipe"] : "inherit",
    env,
    cwd: childCwd,
  });
  if (app.isPackaged) {
    attachSidecarOutput(sidecar.stdout, "stdout");
    attachSidecarOutput(sidecar.stderr, "stderr");
  }
  sidecar.once("error", (err) => {
    const msg = err instanceof Error ? err.message : String(err);
    launcherLog(`tray backend spawn error: ${msg}`, "error");
    console.error(`${APP_DISPLAY_NAME} tray backend spawn error:`, err);
    console.error(`(See ${launcherLogFilePath()} for history.)`);
    void showFatalLaunchError(
      "Backend Failed to Start",
      `${APP_DISPLAY_NAME} could not start its local backend.`,
      msg,
    );
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
    void showFatalLaunchError(
      "Backend Exited",
      `${APP_DISPLAY_NAME} backend exited unexpectedly.`,
      detail,
    );
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

function payloadResolutionDetails(
  resolution: PayloadResolution,
  activeVersion: string | null,
): string {
  return [
    `Launcher version: ${resolution.launcherManifest.launcher_version}`,
    `Active payload: ${activeVersion ?? "none"}`,
    `Recommended payload: ${resolution.defaultVersion}`,
    `Selected payload: ${resolution.selectedVersion ?? "auto"}`,
    `Target payload: ${resolution.desiredVersion}`,
  ].join("\n");
}

async function applyPayloadResolution(
  resolution: PayloadResolution,
  heading: string,
): Promise<void> {
  if (resolution.syncPlan.action === "switch") {
    if (!useInstalledPayloadVersion(resolution.desiredVersion)) {
      throw new Error(`Payload v${resolution.desiredVersion} is not installed locally`);
    }
    return;
  }

  if (resolution.syncPlan.action === "download") {
    if (isPayloadDirPresent(resolution.payloadManifest)) {
      await downloadAndApplyUpdate(resolution.payloadManifest);
    } else {
      await runDownloadWithProgress(heading, resolution.payloadManifest);
    }
  }
}

async function promptForPayloadRestart(
  resolution: PayloadResolution,
  title: string,
  actionLabel: string,
): Promise<void> {
  const activeVersion = getActiveVersion();
  const res = await dialog.showMessageBox({
    type: "info",
    title,
    message: `${APP_DISPLAY_NAME} will restart to use payload v${resolution.desiredVersion}.`,
    detail: payloadResolutionDetails(resolution, activeVersion),
    buttons: [actionLabel, "Later"],
    defaultId: 0,
    cancelId: 1,
  });
  if (res.response === 0) {
    app.relaunch();
    app.quit();
  }
}

async function useResolvedPayloadChoice(
  resolution: PayloadResolution,
  rememberSelection: boolean,
): Promise<void> {
  const effectiveResolution: PayloadResolution = rememberSelection
    ? {
        ...resolution,
        desiredSource: "user-selected",
        selectedVersion: resolution.desiredVersion,
      }
    : {
        ...resolution,
        desiredSource: "default",
        selectedVersion: null,
      };

  if (rememberSelection) {
    setSelectedVersion(effectiveResolution.desiredVersion);
  } else {
    clearSelectedVersion();
  }

  if (effectiveResolution.syncPlan.action === "none") {
    latestPayloadResolution = effectiveResolution;
    new Notification({
      title: APP_DISPLAY_NAME,
      body: rememberSelection
        ? `Payload v${effectiveResolution.desiredVersion} is already active`
        : `Recommended payload v${effectiveResolution.desiredVersion} is already active`,
    }).show();
    return;
  }

  await applyPayloadResolution(effectiveResolution, `Downloading ${APP_DISPLAY_NAME} core`);
  latestPayloadResolution = effectiveResolution;
  await promptForPayloadRestart(
    effectiveResolution,
    "Restart Required",
    "Restart Now",
  );
}

async function switchToRecommendedPayload(): Promise<void> {
  const resolution = await resolvePayloadRelease({
    timeoutMs: manifestFetchTimeoutMs(),
    ignoreSelectedVersion: true,
  });
  if (resolution === null) {
    throw new Error(lastManifestCheckFailure ?? "Failed to resolve recommended payload");
  }
  await useResolvedPayloadChoice(resolution, false);
}

async function switchToPayloadVersion(version: string): Promise<void> {
  const resolution = await resolvePayloadRelease({
    timeoutMs: manifestFetchTimeoutMs(),
    preferredVersion: version,
  });
  if (resolution === null) {
    throw new Error(lastManifestCheckFailure ?? `Failed to resolve payload v${version}`);
  }
  await useResolvedPayloadChoice(resolution, true);
}

async function performBackgroundUpdateCheck() {
  if (!app.isPackaged || isQuitting || mainWindow === null) {
    return;
  }

  try {
    const resolution = await resolvePayloadRelease({ timeoutMs: manifestFetchTimeoutMs() });
    if (resolution === null) {
      return;
    }

    latestPayloadResolution = resolution;
    if (
      resolution.syncPlan.action !== "none"
      && resolution.desiredVersion !== updatePromptShownForVersion
    ) {
      updatePromptShownForVersion = resolution.desiredVersion;
      const activeVersion = getActiveVersion();
      const compatible = orderedCompatiblePayloadVersions(
        resolution.payloadIndex.payloads,
        resolution.launcherManifest.compatible_payloads,
      );
      const offerChooser = payloadChooserOffersMultipleVersions(compatible);
      const buttons = offerChooser
        ? ["Update and Restart", "Choose Version", "Later"]
        : ["Update and Restart", "Later"];

      const res = await dialog.showMessageBox({
        type: "warning",
        title: "Core Update Available",
        message: `${APP_DISPLAY_NAME} can switch to payload v${resolution.desiredVersion} on restart.`,
        detail: payloadResolutionDetails(resolution, activeVersion),
        buttons,
        defaultId: 0,
        cancelId: offerChooser ? 2 : 1,
      });

      const applyUpdateAndRelaunch = async (toApply: PayloadResolution) => {
        await applyPayloadResolution(toApply, `Downloading ${APP_DISPLAY_NAME} update`);
        app.relaunch();
        app.quit();
      };

      if (res.response === 0) {
        try {
          await applyUpdateAndRelaunch(resolution);
        } catch (err) {
          const msg = err instanceof Error ? err.message : String(err);
          launcherLog(`background update apply failed: ${msg}`, "error");
          await showFatalLaunchError(
            "Core Repair Failed",
            `Failed to repair ${APP_DISPLAY_NAME} core.`,
            msg,
          );
        }
      } else if (offerChooser && res.response === 1) {
        const picked = await promptPayloadVersionChoice({
          versions: compatible,
          recommended: resolution.defaultVersion,
          title: "Choose Payload Version",
        });
        if (picked === null) {
          return;
        }
        try {
          const chosenResolution = await resolvePayloadRelease({
            timeoutMs: manifestFetchTimeoutMs(),
            preferredVersion: picked,
          });
          if (chosenResolution === null) {
            throw new Error(lastManifestCheckFailure ?? "Failed to resolve selected payload");
          }
          latestPayloadResolution = chosenResolution;
          await applyUpdateAndRelaunch(chosenResolution);
        } catch (err) {
          const msg = err instanceof Error ? err.message : String(err);
          launcherLog(`background update apply failed: ${msg}`, "error");
          await showFatalLaunchError(
            "Core Repair Failed",
            `Failed to repair ${APP_DISPLAY_NAME} core.`,
            msg,
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

    const activePayloadVersion = getActiveVersion();
    const selectedPayloadVersion = getSelectedVersion();
    const installedPayloadVersions = listInstalledPayloadVersions();
    const payloadResolution = latestPayloadResolution;
    const compatiblePayloadVersions = payloadResolution === null
      ? []
      : orderedCompatiblePayloadVersions(
          payloadResolution.payloadIndex.payloads,
          payloadResolution.launcherManifest.compatible_payloads,
        );
    const installablePayloadVersions = installableOnlyPayloadVersions(
      compatiblePayloadVersions,
      installedPayloadVersions,
    );
    const payloadSubmenu: Electron.MenuItemConstructorOptions[] = payloadResolution === null
      ? [
          { label: `Active: ${activePayloadVersion ?? "none"}`, enabled: false },
          { label: "Payload metadata unavailable", enabled: false },
        ]
      : [
          { label: `Launcher: v${payloadResolution.launcherManifest.launcher_version}`, enabled: false },
          { label: `Active: ${activePayloadVersion ?? "none"}`, enabled: false },
          { label: `Recommended: ${payloadResolution.defaultVersion}`, enabled: false },
          { label: `Selected: ${selectedPayloadVersion ?? "auto"}`, enabled: false },
          { type: "separator" },
          {
            label: `Use Recommended (${payloadResolution.defaultVersion})`,
            type: "checkbox" as const,
            checked: selectedPayloadVersion === null,
            click: () => {
              void switchToRecommendedPayload().catch((err) => {
                const message = err instanceof Error ? err.message : String(err);
                new Notification({ title: APP_DISPLAY_NAME, body: `Payload switch failed: ${message}` }).show();
              });
            },
          },
          ...installedPayloadVersions
            .filter((version) => compatiblePayloadVersions.includes(version))
            .map((version) => ({
              label: `Use Installed v${version}`,
              type: "checkbox" as const,
              checked: selectedPayloadVersion === version,
              click: () => {
                void switchToPayloadVersion(version).catch((err) => {
                  const message = err instanceof Error ? err.message : String(err);
                  new Notification({ title: APP_DISPLAY_NAME, body: `Payload switch failed: ${message}` }).show();
                });
              },
            })),
          ...(installablePayloadVersions.length > 0
            ? [
                { type: "separator" as const },
                {
                  label: "Install Compatible Version",
                  submenu: installablePayloadVersions.map((version) => ({
                    label: `Install v${version}`,
                    click: () => {
                      void switchToPayloadVersion(version).catch((err) => {
                        const message = err instanceof Error ? err.message : String(err);
                        new Notification({ title: APP_DISPLAY_NAME, body: `Payload install failed: ${message}` }).show();
                      });
                    },
                  })),
                },
              ]
            : []),
        ];

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
      { label: "Payload", submenu: payloadSubmenu },
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

/** Modal picker for a compatible payload version (install/update flow; does not persist selection). */
function promptPayloadVersionChoice(options: {
  versions: string[];
  recommended: string;
  title: string;
}): Promise<string | null> {
  const { versions, recommended, title } = options;
  if (versions.length === 0) {
    return Promise.resolve(null);
  }
  return new Promise((resolve) => {
    let settled = false;
    const finish = (value: string | null) => {
      if (settled) {
        return;
      }
      settled = true;
      resolve(value);
    };

    const win = new BrowserWindow({
      width: 440,
      height: 220,
      show: false,
      center: true,
      resizable: false,
      minimizable: false,
      maximizable: false,
      title,
      modal: true,
      parent: mainWindow ?? undefined,
      icon: getAppIconPath(),
      webPreferences: {
        preload: path.join(__dirname, "payload_chooser_preload.js"),
        contextIsolation: true,
        nodeIntegration: false,
        sandbox: true,
      },
    });

    const versionOptions = versions
      .map((v) => {
        const sel = v === recommended ? " selected" : "";
        const escaped = escapeHtmlAttr(v);
        return `<option value="${escaped}"${sel}>${escaped}</option>`;
      })
      .join("");

    const html = `<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>${escapeHtmlAttr(title)}</title>
<style>
  body { margin:0; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; padding:16px 20px; background:#f5f5f5; color:#111; }
  h1 { font-size:15px; font-weight:600; margin:0 0 12px; }
  label { display:block; font-size:13px; margin-bottom:6px; color:#555; }
  select { width:100%; padding:8px 10px; font-size:14px; border-radius:6px; border:1px solid #ccc; box-sizing:border-box; }
  .row { display:flex; gap:10px; justify-content:flex-end; margin-top:16px; }
  button { padding:8px 16px; font-size:14px; border-radius:6px; border:1px solid #ccc; background:#fff; cursor:pointer; }
  button.primary { background:#2563eb; color:#fff; border-color:#2563eb; }
</style>
</head>
<body>
  <h1>${escapeHtmlAttr(title)}</h1>
  <label for="payloadVersion">Payload version</label>
  <select id="payloadVersion">${versionOptions}</select>
  <div class="row">
    <button type="button" id="cancelBtn">Cancel</button>
    <button type="button" class="primary" id="okBtn">OK</button>
  </div>
  <script>
    const api = window.taskclfPayloadChooser;
    if (!api) {
      document.body.innerText = "Payload chooser unavailable.";
    } else {
      const sel = document.getElementById("payloadVersion");
      document.getElementById("okBtn").addEventListener("click", () => {
        api.ok(sel.value);
      });
      document.getElementById("cancelBtn").addEventListener("click", () => {
        api.cancel();
      });
    }
  </script>
</body>
</html>`;

    function cleanup() {
      ipcMain.removeListener("taskclf-payload-chooser-submit", onSubmit);
      ipcMain.removeListener("taskclf-payload-chooser-cancel", onCancel);
    }

    function onSubmit(_e: Electron.IpcMainEvent, version: string) {
      cleanup();
      finish(version);
      if (!win.isDestroyed()) {
        win.destroy();
      }
    }

    function onCancel() {
      cleanup();
      finish(null);
      if (!win.isDestroyed()) {
        win.destroy();
      }
    }

    ipcMain.on("taskclf-payload-chooser-submit", onSubmit);
    ipcMain.on("taskclf-payload-chooser-cancel", onCancel);

    win.once("closed", () => {
      cleanup();
      finish(null);
    });

    win.loadURL(`data:text/html;charset=utf-8,${encodeURIComponent(html)}`);
    win.once("ready-to-show", () => {
      win.show();
    });
  });
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

type ProgressWindowState = {
  heading?: string;
  detail?: string;
  percent?: number | null;
};

function progressWindowPageHtml(heading: string, detail = ""): string {
  const h = escapeHtmlAttr(heading);
  const d = escapeHtmlAttr(detail);
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
  <h1 id="heading">${h}</h1>
  <div id="barwrap" class="indeterminate"><div id="bar"></div></div>
  <p id="line2">${d}</p>
</body>
</html>`;
}

function createProgressWindow(heading: string, detail = ""): BrowserWindow {
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
    `data:text/html;charset=utf-8,${encodeURIComponent(progressWindowPageHtml(heading, detail))}`,
  );
  win.once("ready-to-show", () => {
    win.show();
  });
  return win;
}

async function waitForWindowDocument(win: BrowserWindow): Promise<void> {
  if (win.isDestroyed() || !win.webContents.isLoadingMainFrame()) {
    return;
  }
  await new Promise<void>((resolve) => {
    const done = () => {
      resolve();
    };
    win.webContents.once("did-finish-load", done);
    win.once("closed", done);
  });
}

async function setProgressWindowState(
  win: BrowserWindow,
  state: ProgressWindowState,
): Promise<void> {
  if (win.isDestroyed()) {
    return;
  }
  await waitForWindowDocument(win);
  if (win.isDestroyed()) {
    return;
  }
  const pct = state.percent;
  const heading = state.heading;
  const detail = state.detail;
  const js = `(function(){
    var p = ${pct === null || pct === undefined ? "null" : String(Math.round(pct))};
    var heading = ${heading === undefined ? "null" : JSON.stringify(heading)};
    var line2 = ${detail === undefined ? "null" : JSON.stringify(detail)};
    var title = document.getElementById("heading");
    var bar = document.getElementById("bar");
    var wrap = document.getElementById("barwrap");
    if (title && heading !== null) {
      title.textContent = heading;
      document.title = heading;
    }
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
    if (el && line2 !== null) { el.textContent = line2; }
  })();`;
  await win.webContents.executeJavaScript(js, true);
}

async function applyProgressToWindow(win: BrowserWindow, event: UpdateProgressEvent): Promise<void> {
  await setProgressWindowState(win, {
    detail: formatProgressLine(event),
    percent: event.percent ?? null,
  });
}

function closeProgressWindow(win: BrowserWindow | null): void {
  if (win !== null && !win.isDestroyed()) {
    win.close();
  }
}

async function runDownloadWithProgress(heading: string, manifest: Manifest): Promise<void> {
  const win = createProgressWindow(heading, "Connecting…");
  try {
    await downloadAndApplyUpdate(manifest, {
      onProgress: (e) => applyProgressToWindow(win, e),
    });
  } finally {
    closeProgressWindow(win);
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
  let startupWindow: BrowserWindow | null = null;

  const showStartupStatus = async (detail: string): Promise<void> => {
    if (!app.isPackaged) {
      return;
    }
    const heading = `Starting ${APP_DISPLAY_NAME}`;
    if (startupWindow === null || startupWindow.isDestroyed()) {
      startupWindow = createProgressWindow(heading, detail);
      return;
    }
    await setProgressWindowState(startupWindow, {
      heading,
      detail,
      percent: null,
    });
  };

  const hideStartupStatus = (): void => {
    closeProgressWindow(startupWindow);
    startupWindow = null;
  };

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
    const launcherVersion = app.getVersion();
    const activePayloadVersion = getActiveVersion();
    await showStartupStatus("Checking for core components…");
    const resolution = await resolvePayloadRelease({ timeoutMs: manifestFetchTimeoutMs() });
    if (resolution === null) {
      hideStartupStatus();
      const fallbackVersion = activateBestLocalPayloadFallback();
      if (fallbackVersion !== null) {
        launcherLog(
          `release metadata unavailable; falling back to local payload v${fallbackVersion}: `
          + `${lastManifestCheckFailure ?? "unknown error"}`,
          "error",
        );
      } else {
        const reason = lastManifestCheckFailure
          ? `Release metadata fetch failed: ${lastManifestCheckFailure}`
          : `No payload metadata was returned for launcher v${launcherVersion}.`;
        launcherLog(`launcher payload manifest unavailable: ${reason}`, "error");
        await showFatalLaunchError(
          "Core Download Failed",
          `${APP_DISPLAY_NAME} could not resolve a payload for launcher v${launcherVersion}.`,
          reason,
        );
        return;
      }
    } else {
      latestPayloadResolution = resolution;
      if (resolution.syncPlan.action === "switch") {
        hideStartupStatus();
        if (!useInstalledPayloadVersion(resolution.desiredVersion)) {
          await showFatalLaunchError(
            "Core Switch Failed",
            `${APP_DISPLAY_NAME} could not activate payload v${resolution.desiredVersion}.`,
            "The desired payload is not installed locally.",
          );
          return;
        }
        if (activePayloadVersion !== null && activePayloadVersion !== resolution.desiredVersion) {
          launcherLog(
            `switching active payload from v${activePayloadVersion} to v${resolution.desiredVersion}`,
            "info",
          );
        }
      } else if (resolution.syncPlan.action === "download") {
        hideStartupStatus();
        const actionLabel = activePayloadVersion === null ? "Download and Start" : "Update and Start";
        const message = activePayloadVersion === null
          ? `${APP_DISPLAY_NAME} needs to download its core components before starting.`
          : `${APP_DISPLAY_NAME} needs to update its local core before starting.`;
        const compatible = orderedCompatiblePayloadVersions(
          resolution.payloadIndex.payloads,
          resolution.launcherManifest.compatible_payloads,
        );
        const offerChooser = payloadChooserOffersMultipleVersions(compatible);
        const buttons = offerChooser
          ? [actionLabel, "Choose Version", "Quit"]
          : [actionLabel, "Quit"];
        const res = await dialog.showMessageBox({
          type: "info",
          title: activePayloadVersion === null ? "Initial Setup" : "Core Update Required",
          message,
          detail: payloadResolutionDetails(resolution, activePayloadVersion),
          buttons,
          defaultId: 0,
          cancelId: offerChooser ? 2 : 1,
        });

        const applyDownload = async (toApply: PayloadResolution) => {
          try {
            await applyPayloadResolution(toApply, `Downloading ${APP_DISPLAY_NAME} core`);
          } catch (err) {
            hideStartupStatus();
            const msg = err instanceof Error ? err.message : String(err);
            launcherLog(`launcher payload download/apply failed: ${msg}`, "error");
            await showFatalLaunchError(
              "Core Download Failed",
              `Failed to prepare ${APP_DISPLAY_NAME} core.`,
              msg,
            );
            throw err;
          }
        };

        if (res.response === 0) {
          try {
            await applyDownload(resolution);
          } catch {
            return;
          }
        } else if (offerChooser && res.response === 1) {
          const picked = await promptPayloadVersionChoice({
            versions: compatible,
            recommended: resolution.defaultVersion,
            title: "Choose Payload Version",
          });
          if (picked === null) {
            app.quit();
            return;
          }
          const chosenResolution = await resolvePayloadRelease({
            timeoutMs: manifestFetchTimeoutMs(),
            preferredVersion: picked,
          });
          if (chosenResolution === null) {
            await showFatalLaunchError(
              "Core Download Failed",
              `${APP_DISPLAY_NAME} could not resolve the selected payload.`,
              lastManifestCheckFailure ?? "unknown error",
            );
            return;
          }
          latestPayloadResolution = chosenResolution;
          try {
            await applyDownload(chosenResolution);
          } catch {
            return;
          }
        } else {
          app.quit();
          return;
        }
      } else {
        hideStartupStatus();
      }
    }

    // Check for payload drift in the background on startup.
    performBackgroundUpdateCheck();
  }

  try {
    await showStartupStatus("Starting local backend…");
    spawnSidecar();

    const base = shellUrl();
    await showStartupStatus("Waiting for local UI…");
    await waitForShell(base);
    await showStartupStatus("Opening dashboard…");
    await Promise.all([
      mainWindow.loadURL(base),
      labelWin.loadURL(`${base}?view=label`),
      panelWin.loadURL(`${base}?view=panel`),
    ]);
    hideStartupStatus();
    showWindow();
    syncTimer = setInterval(() => { void syncTrayMenu(); }, 5000);
    updateCheckTimer = setInterval(() => { void performBackgroundUpdateCheck(); }, 60 * 60 * 1000); // 1 hour
    void syncTrayMenu();
  } catch (error) {
    hideStartupStatus();
    const message = error instanceof Error ? error.message : String(error);
    launcherLog(`start failed: ${message}`, "error");
    await showFatalLaunchError(
      "Startup Failed",
      `${APP_DISPLAY_NAME} failed to start.`,
      message,
    );
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
