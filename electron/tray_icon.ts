import path from "node:path";

export const APP_ICON_FILE = "icon.png";
export const MACOS_TRAY_ICON_FILE = "trayTemplate.png";
export const MACOS_TRAY_ICON_RETINA_FILE = "trayTemplate@2x.png";

export function getBuildAssetPath(fileName: string, baseDir: string = __dirname): string {
  return path.join(baseDir, "..", "build", fileName);
}

export function getAppIconPath(baseDir: string = __dirname): string {
  return getBuildAssetPath(APP_ICON_FILE, baseDir);
}

export function getTrayAssetFiles(platform: NodeJS.Platform = process.platform): string[] {
  if (platform === "darwin") {
    return [`build/${MACOS_TRAY_ICON_FILE}`, `build/${MACOS_TRAY_ICON_RETINA_FILE}`];
  }
  return [`build/${APP_ICON_FILE}`];
}

export function getTrayIconAsset(
  baseDir: string = __dirname,
  platform: NodeJS.Platform = process.platform,
): { path: string; template: boolean } {
  if (platform === "darwin") {
    return {
      path: getBuildAssetPath(MACOS_TRAY_ICON_FILE, baseDir),
      template: true,
    };
  }
  return {
    path: getAppIconPath(baseDir),
    template: false,
  };
}
