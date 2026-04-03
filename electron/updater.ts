import { app, net } from "electron";
import fs from "node:fs";
import path from "node:path";
import crypto from "node:crypto";
import AdmZip from "adm-zip";
import {
  manifestUrlForLauncherVersion,
  resolvePayloadSyncPlan,
} from "./update_policy";

export interface Manifest {
  version: string;
  /** Keys are LLVM-style target triples, e.g. x86_64-unknown-linux-gnu */
  platforms: Record<string, {
    url: string;
    sha256: string;
  }>;
}

export type UpdatePhase = "download" | "verify" | "extract";

export interface UpdateProgressEvent {
  phase: UpdatePhase;
  /** Bytes received so far during download */
  receivedBytes?: number;
  /** Total bytes when Content-Length is present */
  totalBytes?: number | null;
  /** 0–100 when known; null if total size is unknown */
  percent?: number | null;
}

export interface DownloadAndApplyOptions {
  onProgress?: (event: UpdateProgressEvent) => void | Promise<void>;
}

export interface CheckForUpdateOptions {
  /** Abort manifest fetch after this many milliseconds; <= 0 disables the timeout. */
  timeoutMs?: number;
}

type UpdaterRequestInit = RequestInit & { bypassCustomProtocolHandlers?: boolean };

async function emitProgress(
  onProgress: DownloadAndApplyOptions["onProgress"],
  event: UpdateProgressEvent,
): Promise<void> {
  if (onProgress) {
    await Promise.resolve(onProgress(event));
  }
}

function parseContentLength(header: string | null): number | null {
  if (!header) return null;
  const n = Number.parseInt(header, 10);
  return Number.isFinite(n) && n >= 0 ? n : null;
}

function percentFromBytes(received: number, total: number | null): number | null {
  if (total === null || total <= 0) return null;
  return Math.min(100, Math.round((received / total) * 100));
}

function describeFetchError(error: unknown): string {
  if (!(error instanceof Error)) {
    return String(error);
  }

  const cause = (error as Error & { cause?: unknown }).cause;
  if (cause === undefined || cause === null) {
    return error.message;
  }
  if (cause instanceof Error) {
    const code = (cause as NodeJS.ErrnoException).code;
    const suffix = code ? `${code}: ${cause.message}` : cause.message;
    return `${error.message} (${suffix})`;
  }
  return `${error.message} (${String(cause)})`;
}

async function updaterFetch(
  input: string,
  purpose: string,
  init?: UpdaterRequestInit,
): Promise<Response> {
  try {
    return await net.fetch(input, {
      cache: "no-store",
      ...init,
    });
  } catch (error) {
    if (error instanceof Error && error.name === "AbortError") {
      throw error;
    }
    throw new Error(`Failed to ${purpose} (${input}): ${describeFetchError(error)}`);
  }
}

/** LLVM-style host target triple for this process; must match scripts/host_target_triple.py */
export function hostTargetTriple(): string {
  const plat = process.platform;
  const arch = process.arch;
  if (plat === "darwin") {
    if (arch === "arm64") return "aarch64-apple-darwin";
    if (arch === "x64") return "x86_64-apple-darwin";
    throw new Error(`unsupported macOS arch: ${arch}`);
  }
  if (plat === "linux") {
    if (arch === "x64") return "x86_64-unknown-linux-gnu";
    if (arch === "arm64") return "aarch64-unknown-linux-gnu";
    if (arch === "ia32") return "i686-unknown-linux-gnu";
    throw new Error(`unsupported Linux arch: ${arch}`);
  }
  if (plat === "win32") {
    if (arch === "x64") return "x86_64-pc-windows-msvc";
    if (arch === "arm64") return "aarch64-pc-windows-msvc";
    if (arch === "ia32") return "i686-pc-windows-msvc";
    throw new Error(`unsupported Windows arch: ${arch}`);
  }
  throw new Error(`unsupported platform: ${plat}`);
}

function getVersionsDir(): string {
  return path.join(app.getPath("userData"), "versions");
}

function getActivePath(): string {
  return path.join(getVersionsDir(), "active.json");
}

export function getActiveVersion(): string | null {
  const activePath = getActivePath();
  if (!fs.existsSync(activePath)) {
    return null;
  }
  try {
    const data = JSON.parse(fs.readFileSync(activePath, "utf-8"));
    return data.version || null;
  } catch {
    return null;
  }
}

function setActiveVersion(version: string): void {
  const activePath = getActivePath();
  fs.writeFileSync(activePath, JSON.stringify({ version }));
}

function getPayloadDir(version: string): string {
  return path.join(getVersionsDir(), `v${version}`);
}

function payloadExecutableName(): string {
  return process.platform === "win32" ? "entry.exe" : "entry";
}

export function getPayloadBackendPath(version: string): string | null {
  const executablePath = path.join(
    getPayloadDir(version),
    "backend",
    payloadExecutableName(),
  );
  if (fs.existsSync(executablePath)) {
    return executablePath;
  }
  return null;
}

/** True if the extracted payload directory for this manifest version already exists (no download needed). */
export function isPayloadDirPresent(manifest: Manifest): boolean {
  return getPayloadBackendPath(manifest.version) !== null;
}

export function getActivePayloadBackendPath(): string | null {
  const version = getActiveVersion();
  if (!version) return null;
  return getPayloadBackendPath(version);
}

export function useInstalledPayloadVersion(version: string): boolean {
  if (getPayloadBackendPath(version) === null) {
    return false;
  }
  setActiveVersion(version);
  return true;
}

/** Set on fetch/parse failures; cleared at the start of each check. For UI / logs only. */
export let lastManifestCheckFailure: string | null = null;

async function fetchWithTimeout(
  input: string,
  timeoutMs?: number,
): Promise<Response> {
  if (timeoutMs === undefined || timeoutMs <= 0) {
    return updaterFetch(input, "fetch manifest");
  }

  const controller = new AbortController();
  const timer = setTimeout(() => {
    controller.abort();
  }, timeoutMs);

  try {
    return await updaterFetch(input, "fetch manifest", { signal: controller.signal });
  } finally {
    clearTimeout(timer);
  }
}

export async function checkForUpdate(
  options?: CheckForUpdateOptions,
): Promise<Manifest | null> {
  const manifestUrl = manifestUrlForLauncherVersion(
    app.getVersion(),
    process.env.TASKCLF_MANIFEST_URL,
  );
  const timeoutMs = options?.timeoutMs;
  lastManifestCheckFailure = null;

  try {
    console.log(`[updater] Checking for updates at ${manifestUrl}`);
    const res = await fetchWithTimeout(manifestUrl, timeoutMs);
    if (!res.ok) {
      lastManifestCheckFailure = `HTTP ${res.status} ${res.statusText} (${manifestUrl})`;
      console.error(`[updater] Failed to fetch manifest: ${res.statusText}`);
      return null;
    }

    let manifest: Manifest;
    try {
      manifest = (await res.json()) as Manifest;
    } catch (parseErr) {
      lastManifestCheckFailure = `Invalid JSON in manifest: ${parseErr instanceof Error ? parseErr.message : String(parseErr)}`;
      console.error("[updater] Manifest JSON parse failed:", parseErr);
      return null;
    }
    const activePayloadPath = getActivePayloadBackendPath();
    const activeVersion = getActiveVersion();
    const syncPlan = resolvePayloadSyncPlan({
      launcherVersion: manifest.version,
      activeVersion,
      activePayloadPresent: activePayloadPath !== null,
      launcherPayloadPresent: getPayloadBackendPath(manifest.version) !== null,
    });

    if (syncPlan.action === "none") {
      console.log(`[updater] Payload ready for launcher v${manifest.version}`);
      return null;
    }

    console.log(`[updater] Payload sync needed: ${syncPlan.reason}`);
    return manifest;
  } catch (error) {
    if (
      timeoutMs !== undefined
      && timeoutMs > 0
      && error instanceof Error
      && error.name === "AbortError"
    ) {
      lastManifestCheckFailure = `Timed out fetching manifest after ${timeoutMs} ms (${manifestUrl})`;
      console.error("[updater] Manifest fetch timed out");
      return null;
    }
    lastManifestCheckFailure = error instanceof Error ? error.message : String(error);
    console.error("[updater] Error checking for update:", error);
    return null;
  }
}

async function streamPayloadToFile(
  downloadRes: Response,
  zipPath: string,
  expectedSha256: string,
  onProgress: DownloadAndApplyOptions["onProgress"],
): Promise<void> {
  if (!downloadRes.body) {
    throw new Error("Download response has no body");
  }

  const totalBytes = parseContentLength(downloadRes.headers.get("content-length"));
  const hash = crypto.createHash("sha256");
  const writeStream = fs.createWriteStream(zipPath);
  const reader = downloadRes.body.getReader();
  let receivedBytes = 0;

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) {
        break;
      }
      if (value === undefined || value.byteLength === 0) {
        continue;
      }
      const buf = Buffer.from(value);
      receivedBytes += buf.length;
      hash.update(buf);
      await new Promise<void>((resolve, reject) => {
        writeStream.write(buf, (err) => {
          if (err) reject(err);
          else resolve();
        });
      });
      await emitProgress(onProgress, {
        phase: "download",
        receivedBytes,
        totalBytes,
        percent: percentFromBytes(receivedBytes, totalBytes),
      });
    }
    await new Promise<void>((resolve, reject) => {
      writeStream.end((err: NodeJS.ErrnoException | null | undefined) => {
        if (err) reject(err);
        else resolve();
      });
    });
  } catch (err) {
    writeStream.destroy();
    try {
      fs.unlinkSync(zipPath);
    } catch {
      // ignore
    }
    throw err;
  }

  const computedHash = hash.digest("hex");
  if (computedHash !== expectedSha256) {
    try {
      fs.unlinkSync(zipPath);
    } catch {
      // ignore
    }
    throw new Error(`Hash mismatch! Expected ${expectedSha256}, got ${computedHash}`);
  }
}

export async function downloadAndApplyUpdate(
  manifest: Manifest,
  options?: DownloadAndApplyOptions,
): Promise<void> {
  const onProgress = options?.onProgress;
  try {
    // Create versions dir if not exists
    const versionsDir = getVersionsDir();
    if (!fs.existsSync(versionsDir)) {
      fs.mkdirSync(versionsDir, { recursive: true });
    }

    const payloadDir = getPayloadDir(manifest.version);
    if (getPayloadBackendPath(manifest.version) !== null) {
      // Already downloaded and executable is present; maybe just not active yet.
      console.log(`[updater] Payload already installed. Switching active version.`);
      setActiveVersion(manifest.version);
      return;
    }

    const triple = hostTargetTriple();
    const platformData = manifest.platforms[triple];
    if (!platformData) {
      throw new Error(`No payload available for target ${triple}`);
    }

    // Download
    const zipPath = path.join(versionsDir, `payload-${manifest.version}.zip`);
    console.log(`[updater] Downloading payload from ${platformData.url}`);

    const downloadRes = await updaterFetch(platformData.url, "download payload");
    if (!downloadRes.ok) {
      throw new Error(`Failed to download payload: ${downloadRes.statusText}`);
    }

    await streamPayloadToFile(downloadRes, zipPath, platformData.sha256, onProgress);

    // Verify Hash (already verified while streaming; emit phase for UI)
    console.log(`[updater] Verifying hash...`);
    await emitProgress(onProgress, { phase: "verify", percent: 100 });

    // Extract
    console.log(`[updater] Extracting payload...`);
    await emitProgress(onProgress, { phase: "extract", percent: null });

    const zip = new AdmZip(zipPath);
    zip.extractAllTo(payloadDir, true);

    // Set executable permissions for Unix
    if (process.platform !== "win32") {
      const execPath = path.join(payloadDir, "backend", "entry");
      if (fs.existsSync(execPath)) {
        fs.chmodSync(execPath, 0o755);
      }
    }

    // Switch active version
    console.log(`[updater] Update complete. Setting active version to ${manifest.version}`);
    setActiveVersion(manifest.version);

    // Cleanup zip
    fs.unlinkSync(zipPath);
  } catch (error) {
    console.error("[updater] Error applying update:", error);
    throw error;
  }
}
