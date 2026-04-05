/**
 * Startup port preflight: detect listeners on the UI port, classify whether they
 * look like taskclf's sidecar, and optionally terminate them before spawn.
 */

import { spawnSync, type SpawnSyncOptionsWithStringEncoding } from "node:child_process";
import { setTimeout as delayMs } from "node:timers/promises";

export type ListenerKind = "taskclf" | "non-taskclf" | "unknown";

export type PortListenerInfo = {
  pid: number;
  commandLine: string;
  kind: ListenerKind;
};

/** Result shape compatible with Node's spawnSync for dependency injection in tests. */
export type SpawnSyncResult = {
  status: number | null;
  stdout: string;
  stderr: string;
};

export type SpawnSyncFn = (
  command: string,
  args: readonly string[],
  options?: SpawnSyncOptionsWithStringEncoding,
) => SpawnSyncResult;

export function defaultSpawnSync(
  command: string,
  args: readonly string[],
  options?: SpawnSyncOptionsWithStringEncoding,
): SpawnSyncResult {
  const r = spawnSync(command, args as string[], {
    ...options,
    encoding: "utf8",
    maxBuffer: 10 * 1024 * 1024,
  });
  return {
    status: r.status,
    stdout: String(r.stdout ?? ""),
    stderr: String(r.stderr ?? ""),
  };
}

/**
 * Classify a process command line as taskclf-owned only when evidence is strong
 * (packaged backend path, Electron userData layout, or dev `python -m taskclf.cli.main tray`).
 */
export function classifyListenerCommandLine(commandLine: string): ListenerKind {
  const s = commandLine.trim();
  if (s.length === 0) {
    return "unknown";
  }
  const lower = s.toLowerCase();

  const hasTraySidecar =
    lower.includes("tray") && (lower.includes("--browser") || lower.includes("--no-tray"));

  if (
    (lower.includes("taskclf-electron") && lower.includes("backend"))
    && (lower.includes("/entry") || lower.includes("\\entry"))
  ) {
    return "taskclf";
  }
  if (lower.includes("backend/entry") || lower.includes("backend\\entry")) {
    if (hasTraySidecar || lower.includes("taskclf")) {
      return "taskclf";
    }
  }
  if (lower.includes("taskclf.app") && hasTraySidecar) {
    return "taskclf";
  }
  if (
    lower.includes("taskclf.cli.main")
    && lower.includes("tray")
    && (lower.includes("--browser") || lower.includes("--no-tray"))
  ) {
    return "taskclf";
  }
  if (lower.includes("taskclf")) {
    return "unknown";
  }
  return "non-taskclf";
}

function parseFirstPidFromLsofT(stdout: string): number | null {
  const lines = stdout.trim().split(/\r?\n/).filter((l) => l.length > 0);
  for (const line of lines) {
    const n = Number.parseInt(line.trim(), 10);
    if (Number.isFinite(n) && n > 0) {
      return n;
    }
  }
  return null;
}

/** Parse `ss -lntp` output for users:(("name",pid=12345,... */
export function parsePidFromSsOutput(text: string): number | null {
  const m = text.match(/pid=(\d+)/);
  if (!m) {
    return null;
  }
  const n = Number.parseInt(m[1]!, 10);
  return Number.isFinite(n) && n > 0 ? n : null;
}

function getListeningPidUnix(
  port: number,
  platform: NodeJS.Platform,
  spawnSyncFn: SpawnSyncFn,
): number | null {
  const lsof = spawnSyncFn("lsof", ["-nP", `-iTCP:${port}`, "-sTCP:LISTEN", "-t"]);
  const pidFromLsof = parseFirstPidFromLsofT(lsof.stdout);
  if (pidFromLsof !== null) {
    return pidFromLsof;
  }
  if (platform === "linux") {
    const ss = spawnSyncFn("ss", ["-lntp", `sport = :${port}`]);
    if (ss.status === 0 && ss.stdout) {
      return parsePidFromSsOutput(ss.stdout);
    }
  }
  return null;
}

function getListeningPidWindows(port: number, spawnSyncFn: SpawnSyncFn): number | null {
  const script =
    `(Get-NetTCPConnection -LocalPort ${port} -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1 -ExpandProperty OwningProcess)`;
  const ps = spawnSyncFn("powershell.exe", ["-NoProfile", "-Command", script]);
  if (ps.status !== 0 && ps.status !== null) {
    return null;
  }
  const raw = ps.stdout.trim();
  if (raw.length === 0) {
    return null;
  }
  const n = Number.parseInt(raw, 10);
  return Number.isFinite(n) && n > 0 ? n : null;
}

function getProcessCommandLine(
  pid: number,
  platform: NodeJS.Platform,
  spawnSyncFn: SpawnSyncFn,
): string {
  if (platform === "win32") {
    const script =
      `(Get-CimInstance Win32_Process -Filter "ProcessId = ${pid}").CommandLine`;
    const ps = spawnSyncFn("powershell.exe", ["-NoProfile", "-Command", script]);
    if (ps.status === 0 && ps.stdout.trim().length > 0) {
      return ps.stdout.trim();
    }
    return "";
  }
  const field = platform === "linux" ? "args=" : "command=";
  const out = spawnSyncFn("ps", ["-p", String(pid), "-ww", "-o", field]);
  if (out.status === 0) {
    return out.stdout.trim();
  }
  return "";
}

/**
 * If nothing is listening on 127.0.0.1:port (TCP), returns null.
 * Otherwise returns PID, best-effort command line, and classification.
 */
export function getPortListenerInfo(
  port: number,
  platform: NodeJS.Platform,
  spawnSyncFn: SpawnSyncFn = defaultSpawnSync,
): PortListenerInfo | null {
  let pid: number | null = null;
  if (platform === "win32") {
    pid = getListeningPidWindows(port, spawnSyncFn);
  } else {
    pid = getListeningPidUnix(port, platform, spawnSyncFn);
  }
  if (pid === null) {
    return null;
  }
  const commandLine = getProcessCommandLine(pid, platform, spawnSyncFn);
  const kind = classifyListenerCommandLine(commandLine);
  return { pid, commandLine, kind };
}

const DEFAULT_KILL_WAIT_MS = 10_000;
const POLL_MS = 200;

export function isPortInUse(
  port: number,
  platform: NodeJS.Platform,
  spawnSyncFn: SpawnSyncFn = defaultSpawnSync,
): boolean {
  return getPortListenerInfo(port, platform, spawnSyncFn) !== null;
}

/**
 * Send SIGTERM (Unix) or taskkill (Windows), then poll until the port is free or timeout.
 */
export async function killPidAndWaitForPortFree(
  port: number,
  pid: number,
  platform: NodeJS.Platform,
  spawnSyncFn: SpawnSyncFn = defaultSpawnSync,
  timeoutMs = DEFAULT_KILL_WAIT_MS,
): Promise<boolean> {
  if (platform === "win32") {
    spawnSyncFn("taskkill.exe", ["/PID", String(pid), "/T"]);
  } else {
    spawnSyncFn("kill", ["-TERM", String(pid)]);
  }

  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    if (!isPortInUse(port, platform, spawnSyncFn)) {
      return true;
    }
    await delayMs(POLL_MS);
  }

  if (platform === "win32") {
    spawnSyncFn("taskkill.exe", ["/PID", String(pid), "/T", "/F"]);
  } else {
    spawnSyncFn("kill", ["-KILL", String(pid)]);
  }

  const hardDeadline = Date.now() + 3000;
  while (Date.now() < hardDeadline) {
    if (!isPortInUse(port, platform, spawnSyncFn)) {
      return true;
    }
    await delayMs(POLL_MS);
  }
  return !isPortInUse(port, platform, spawnSyncFn);
}
