import { app, net } from "electron";
import fs from "node:fs";
import path from "node:path";
import crypto from "node:crypto";
import AdmZip from "adm-zip";
import {
  isPayloadVersionCompatible,
  manifestUrlForLauncherVersion,
  resolvePayloadSyncPlan,
  selectLatestCompatiblePayloadVersion,
  type CompatiblePayloadRange,
  type PayloadSyncPlan,
} from "./update_policy";

export interface PayloadPlatformData {
  url: string;
  sha256: string;
}

export interface PayloadManifest {
  kind?: "payload";
  schema_version?: number;
  version: string;
  /** Keys are LLVM-style target triples, e.g. x86_64-unknown-linux-gnu */
  platforms: Record<string, PayloadPlatformData>;
}

export type Manifest = PayloadManifest;

export interface LauncherManifest {
  kind?: "launcher";
  schema_version?: number;
  launcher_version: string;
  payload_index_url: string;
  default_payload_selection?: {
    strategy: "latest-compatible";
  };
  compatible_payloads: CompatiblePayloadRange;
}

export interface PayloadIndexEntry {
  version: string;
  manifest_url: string;
}

export interface PayloadIndex {
  kind?: "payload-index";
  schema_version?: number;
  generated_at?: string;
  payloads: PayloadIndexEntry[];
}

export type DesiredPayloadSource = "default" | "user-selected";

export interface PayloadResolution {
  launcherManifest: LauncherManifest;
  payloadIndex: PayloadIndex;
  payloadManifest: PayloadManifest;
  defaultVersion: string;
  desiredVersion: string;
  desiredSource: DesiredPayloadSource;
  selectedVersion: string | null;
  syncPlan: PayloadSyncPlan;
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
  preferredVersion?: string;
  ignoreSelectedVersion?: boolean;
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

function ensureVersionsDir(): void {
  const versionsDir = getVersionsDir();
  if (!fs.existsSync(versionsDir)) {
    fs.mkdirSync(versionsDir, { recursive: true });
  }
}

function getActivePath(): string {
  return path.join(getVersionsDir(), "active.json");
}

function getSelectedPath(): string {
  return path.join(getVersionsDir(), "selected.json");
}

function readStoredVersion(filePath: string): string | null {
  if (!fs.existsSync(filePath)) {
    return null;
  }
  try {
    const data = JSON.parse(fs.readFileSync(filePath, "utf-8"));
    return data.version || null;
  } catch {
    return null;
  }
}

function writeStoredVersion(filePath: string, version: string): void {
  ensureVersionsDir();
  fs.writeFileSync(filePath, JSON.stringify({ version }));
}

function clearStoredVersion(filePath: string): void {
  if (fs.existsSync(filePath)) {
    fs.unlinkSync(filePath);
  }
}

export function getActiveVersion(): string | null {
  return readStoredVersion(getActivePath());
}

function setActiveVersion(version: string): void {
  writeStoredVersion(getActivePath(), version);
}

export function getSelectedVersion(): string | null {
  return readStoredVersion(getSelectedPath());
}

export function setSelectedVersion(version: string): void {
  writeStoredVersion(getSelectedPath(), version);
}

export function clearSelectedVersion(): void {
  clearStoredVersion(getSelectedPath());
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

export function listInstalledPayloadVersions(): string[] {
  const versionsDir = getVersionsDir();
  if (!fs.existsSync(versionsDir)) {
    return [];
  }

  return fs.readdirSync(versionsDir)
    .filter((entry) => entry.startsWith("v"))
    .map((entry) => entry.slice(1))
    .filter((version) => version.length > 0 && getPayloadBackendPath(version) !== null)
    .sort((left, right) => right.localeCompare(left, undefined, { numeric: true }));
}

/** True if the extracted payload directory for this manifest version already exists (no download needed). */
export function isPayloadDirPresent(manifest: PayloadManifest): boolean {
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
  purpose: string,
  timeoutMs?: number,
): Promise<Response> {
  if (timeoutMs === undefined || timeoutMs <= 0) {
    return updaterFetch(input, purpose);
  }

  const controller = new AbortController();
  const timer = setTimeout(() => {
    controller.abort();
  }, timeoutMs);

  try {
    return await updaterFetch(input, purpose, { signal: controller.signal });
  } finally {
    clearTimeout(timer);
  }
}

async function fetchJsonWithTimeout<T>(
  input: string,
  purpose: string,
  timeoutMs?: number,
): Promise<T> {
  const res = await fetchWithTimeout(input, purpose, timeoutMs);
  if (!res.ok) {
    throw new Error(`HTTP ${res.status} ${res.statusText} (${input})`);
  }

  try {
    return (await res.json()) as T;
  } catch (parseErr) {
    throw new Error(
      `Invalid JSON in ${purpose}: ${parseErr instanceof Error ? parseErr.message : String(parseErr)}`,
    );
  }
}

function validateLauncherManifest(manifest: LauncherManifest, expectedVersion: string): void {
  if (manifest.launcher_version !== expectedVersion) {
    throw new Error(
      `Launcher manifest version mismatch: expected v${expectedVersion}, got v${manifest.launcher_version}`,
    );
  }
  if (!manifest.payload_index_url) {
    throw new Error("Launcher manifest missing payload_index_url");
  }
  if (!manifest.compatible_payloads) {
    throw new Error("Launcher manifest missing compatible_payloads");
  }
}

function findPayloadIndexEntry(payloadIndex: PayloadIndex, version: string): PayloadIndexEntry | null {
  return payloadIndex.payloads.find((entry) => entry.version === version) ?? null;
}

function resolveDesiredPayloadVersion(
  launcherManifest: LauncherManifest,
  payloadIndex: PayloadIndex,
  preferredVersion?: string,
  ignoreSelectedVersion?: boolean,
): {
  defaultVersion: string;
  version: string;
  source: DesiredPayloadSource;
  selectedVersion: string | null;
} {
  const selectedVersion = ignoreSelectedVersion ? null : getSelectedVersion();
  const availableVersions = payloadIndex.payloads.map((entry) => entry.version);
  const latestCompatible = selectLatestCompatiblePayloadVersion({
    compatiblePayloads: launcherManifest.compatible_payloads,
    availableVersions,
  });
  if (latestCompatible === null) {
    throw new Error(
      `No payload release satisfies launcher compatibility range `
      + `[${launcherManifest.compatible_payloads.min_version_inclusive}, `
      + `${launcherManifest.compatible_payloads.max_version_exclusive})`,
    );
  }

  if (preferredVersion !== undefined) {
    if (!availableVersions.includes(preferredVersion)) {
      throw new Error(`Payload v${preferredVersion} is not present in the payload index`);
    }
    if (!isPayloadVersionCompatible(preferredVersion, launcherManifest.compatible_payloads)) {
      throw new Error(`Payload v${preferredVersion} is not compatible with launcher v${launcherManifest.launcher_version}`);
    }
    return {
      defaultVersion: latestCompatible,
      version: preferredVersion,
      source: "user-selected",
      selectedVersion,
    };
  }

  if (
    selectedVersion !== null
    && availableVersions.includes(selectedVersion)
    && isPayloadVersionCompatible(selectedVersion, launcherManifest.compatible_payloads)
  ) {
    return {
      defaultVersion: latestCompatible,
      version: selectedVersion,
      source: "user-selected",
      selectedVersion,
    };
  }

  return {
    defaultVersion: latestCompatible,
    version: latestCompatible,
    source: "default",
    selectedVersion,
  };
}

export async function resolvePayloadRelease(
  options?: CheckForUpdateOptions,
): Promise<PayloadResolution | null> {
  const launcherManifestUrl = manifestUrlForLauncherVersion(
    app.getVersion(),
    process.env.TASKCLF_MANIFEST_URL,
  );
  const timeoutMs = options?.timeoutMs;
  lastManifestCheckFailure = null;

  try {
    console.log(`[updater] Checking launcher manifest at ${launcherManifestUrl}`);
    const launcherManifest = await fetchJsonWithTimeout<LauncherManifest>(
      launcherManifestUrl,
      "launcher manifest",
      timeoutMs,
    );
    validateLauncherManifest(launcherManifest, app.getVersion());

    const payloadIndex = await fetchJsonWithTimeout<PayloadIndex>(
      launcherManifest.payload_index_url,
      "payload index",
      timeoutMs,
    );
    const desired = resolveDesiredPayloadVersion(
      launcherManifest,
      payloadIndex,
      options?.preferredVersion,
      options?.ignoreSelectedVersion,
    );
    const payloadIndexEntry = findPayloadIndexEntry(payloadIndex, desired.version);
    if (payloadIndexEntry === null) {
      throw new Error(`Payload index entry missing for v${desired.version}`);
    }

    const payloadManifest = await fetchJsonWithTimeout<PayloadManifest>(
      payloadIndexEntry.manifest_url,
      `payload manifest for v${desired.version}`,
      timeoutMs,
    );
    const activePayloadPath = getActivePayloadBackendPath();
    const activeVersion = getActiveVersion();
    const syncPlan = resolvePayloadSyncPlan({
      desiredVersion: payloadManifest.version,
      activeVersion,
      activePayloadPresent: activePayloadPath !== null,
      desiredPayloadPresent: getPayloadBackendPath(payloadManifest.version) !== null,
    });

    return {
      launcherManifest,
      payloadIndex,
      payloadManifest,
      defaultVersion: desired.defaultVersion,
      desiredVersion: payloadManifest.version,
      desiredSource: desired.source,
      selectedVersion: desired.selectedVersion,
      syncPlan,
    };
  } catch (error) {
    if (
      timeoutMs !== undefined
      && timeoutMs > 0
      && error instanceof Error
      && error.name === "AbortError"
    ) {
      lastManifestCheckFailure = `Timed out fetching release metadata after ${timeoutMs} ms`;
      console.error("[updater] Release metadata fetch timed out");
      return null;
    }
    lastManifestCheckFailure = error instanceof Error ? error.message : String(error);
    console.error("[updater] Error resolving payload release:", error);
    return null;
  }
}

export async function checkForUpdate(
  options?: CheckForUpdateOptions,
): Promise<PayloadResolution | null> {
  const resolution = await resolvePayloadRelease(options);
  if (resolution === null) {
    return null;
  }

  if (resolution.syncPlan.action === "none") {
    console.log(
      `[updater] Payload ready for launcher v${resolution.launcherManifest.launcher_version}`
      + ` using payload v${resolution.desiredVersion}`,
    );
    return null;
  }

  console.log(`[updater] Payload sync needed: ${resolution.syncPlan.reason}`);
  return resolution;
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
  manifest: PayloadManifest,
  options?: DownloadAndApplyOptions,
): Promise<void> {
  const onProgress = options?.onProgress;
  try {
    ensureVersionsDir();
    const versionsDir = getVersionsDir();

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
