export interface PayloadSyncPlanInput {
  launcherVersion: string;
  activeVersion: string | null;
  activePayloadPresent: boolean;
  launcherPayloadPresent: boolean;
}

export type PayloadSyncAction = "none" | "switch" | "download";

export interface PayloadSyncPlan {
  action: PayloadSyncAction;
  reason: string;
}

export function manifestUrlForLauncherVersion(
  version: string,
  overrideUrl?: string,
): string {
  if (overrideUrl && overrideUrl.length > 0) {
    return overrideUrl;
  }
  return `https://github.com/fruitiecutiepie/taskclf/releases/download/launcher-v${version}/manifest.json`;
}

export function resolvePayloadSyncPlan(
  input: PayloadSyncPlanInput,
): PayloadSyncPlan {
  const {
    launcherVersion,
    activeVersion,
    activePayloadPresent,
    launcherPayloadPresent,
  } = input;

  if (activeVersion === launcherVersion && activePayloadPresent) {
    return {
      action: "none",
      reason: `payload v${launcherVersion} already matches the launcher`,
    };
  }

  if (launcherPayloadPresent) {
    return {
      action: "switch",
      reason: `switching from payload v${activeVersion ?? "none"} to launcher-matched payload v${launcherVersion}`,
    };
  }

  return {
    action: "download",
    reason: `launcher v${launcherVersion} requires payload v${launcherVersion}, but the payload is not installed locally`,
  };
}
