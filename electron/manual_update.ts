import type { LauncherResolution, PayloadResolution } from "./updater";

export type ManualUpdateState = {
  launcher: LauncherResolution;
  payload: PayloadResolution;
  activePayloadVersion: string | null;
  selectedPayloadVersion: string | null;
  selectionIgnoredNote: string | null;
  compatiblePayloadVersions: string[];
};

export type ManualUpdateChoice = {
  launcher: boolean;
  core: boolean;
  coreVersion: string | null;
};

export function launcherResolutionSummary(
  resolution: LauncherResolution,
): string {
  if (!resolution.updateAvailable) {
    return `Launcher is current at v${resolution.currentVersion}.`;
  }
  if (resolution.platformData === null) {
    return `Launcher v${resolution.latestVersion} is available, but no installer is published for this platform.`;
  }
  return `Launcher v${resolution.latestVersion} is available (current v${resolution.currentVersion}).`;
}

export function launcherResolutionIsActionable(
  resolution: LauncherResolution,
): boolean {
  return resolution.updateAvailable && resolution.platformData !== null;
}

export function manualUpdateDialogDetails(state: ManualUpdateState): string {
  const lines = [
    `Launcher current: v${state.launcher.currentVersion}`,
    `Launcher latest: v${state.launcher.latestVersion}`,
    `Core active: ${state.activePayloadVersion ?? "none"}`,
    `Core recommended: ${state.payload.defaultVersion}`,
    `Core selected: ${state.selectedPayloadVersion ?? state.payload.selectedVersion ?? "auto"}`,
    `Core target: ${state.payload.desiredVersion}`,
    "",
    launcherResolutionSummary(state.launcher),
  ];
  if (state.selectionIgnoredNote) {
    lines.push("", state.selectionIgnoredNote);
  }
  return lines.join("\n");
}

export function combinedUpdateStateHasAnyUpdate(state: ManualUpdateState): boolean {
  return launcherResolutionIsActionable(state.launcher)
    || state.payload.syncPlan.action !== "none";
}
