import { app } from "electron";
import fs from "node:fs";
import path from "node:path";
import crypto from "node:crypto";
import AdmZip from "adm-zip";

export interface Manifest {
  version: string;
  /** Keys are LLVM-style target triples, e.g. x86_64-unknown-linux-gnu */
  platforms: Record<string, {
    url: string;
    sha256: string;
  }>;
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

export function getActivePayloadBackendPath(): string | null {
  const version = getActiveVersion();
  if (!version) return null;

  const payloadDir = getPayloadDir(version);
  const name = process.platform === "win32" ? "entry.exe" : "entry";
  const executablePath = path.join(payloadDir, "backend", name);

  if (fs.existsSync(executablePath)) {
    return executablePath;
  }
  return null;
}

export async function checkForUpdate(): Promise<Manifest | null> {
  const manifestUrl = process.env.TASKCLF_MANIFEST_URL || "https://github.com/fruitiecutiepie/taskclf/releases/latest/download/manifest.json";

  try {
    console.log(`[updater] Checking for updates at ${manifestUrl}`);
    const res = await fetch(manifestUrl);
    if (!res.ok) {
      console.error(`[updater] Failed to fetch manifest: ${res.statusText}`);
      return null;
    }

    const manifest: Manifest = await res.json() as Manifest;
    const activeVersion = getActiveVersion();

    if (activeVersion === manifest.version) {
      console.log(`[updater] Up to date (version ${activeVersion})`);
      return null;
    }

    console.log(`[updater] Update found: ${manifest.version} (current: ${activeVersion || 'none'})`);
    return manifest;
  } catch (error) {
    console.error("[updater] Error checking for update:", error);
    return null;
  }
}

export async function downloadAndApplyUpdate(manifest: Manifest): Promise<void> {
  try {
    // Create versions dir if not exists
    const versionsDir = getVersionsDir();
    if (!fs.existsSync(versionsDir)) {
      fs.mkdirSync(versionsDir, { recursive: true });
    }

    const payloadDir = getPayloadDir(manifest.version);
    if (fs.existsSync(payloadDir)) {
       // Already downloaded but maybe not active?
       console.log(`[updater] Payload dir already exists. Switching active version.`);
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

    const downloadRes = await fetch(platformData.url);
    if (!downloadRes.ok) {
      throw new Error(`Failed to download payload: ${downloadRes.statusText}`);
    }

    const arrayBuffer = await downloadRes.arrayBuffer();
    const buffer = Buffer.from(arrayBuffer);
    fs.writeFileSync(zipPath, buffer);

    // Verify Hash
    console.log(`[updater] Verifying hash...`);
    const hash = crypto.createHash("sha256");
    hash.update(buffer);
    const computedHash = hash.digest("hex");

    if (computedHash !== platformData.sha256) {
      fs.unlinkSync(zipPath);
      throw new Error(`Hash mismatch! Expected ${platformData.sha256}, got ${computedHash}`);
    }

    // Extract
    console.log(`[updater] Extracting payload...`);
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
