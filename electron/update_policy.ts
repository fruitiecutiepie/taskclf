export interface CompatiblePayloadRange {
  min_version_inclusive: string;
  max_version_exclusive: string;
}

export interface PayloadSelectionInput {
  compatiblePayloads: CompatiblePayloadRange;
  availableVersions: string[];
}

export interface PayloadSyncPlanInput {
  desiredVersion: string;
  activeVersion: string | null;
  activePayloadPresent: boolean;
  desiredPayloadPresent: boolean;
}

export type PayloadSyncAction = "none" | "switch" | "download";

export interface PayloadSyncPlan {
  action: PayloadSyncAction;
  reason: string;
}

function parseVersion(version: string): [number, number, number] {
  const match = /^(\d+)\.(\d+)\.(\d+)$/.exec(version);
  if (!match) {
    throw new Error(`Unsupported version format: ${version}`);
  }
  return [
    Number.parseInt(match[1], 10),
    Number.parseInt(match[2], 10),
    Number.parseInt(match[3], 10),
  ];
}

export function compareVersions(left: string, right: string): number {
  const a = parseVersion(left);
  const b = parseVersion(right);
  for (let i = 0; i < a.length; i += 1) {
    if (a[i] === b[i]) {
      continue;
    }
    return a[i] < b[i] ? -1 : 1;
  }
  return 0;
}

export function isPayloadVersionCompatible(
  version: string,
  compatiblePayloads: CompatiblePayloadRange,
): boolean {
  return compareVersions(version, compatiblePayloads.min_version_inclusive) >= 0
    && compareVersions(version, compatiblePayloads.max_version_exclusive) < 0;
}

export function selectLatestCompatiblePayloadVersion(
  input: PayloadSelectionInput,
): string | null {
  const compatibleVersions = input.availableVersions
    .filter((version) => isPayloadVersionCompatible(version, input.compatiblePayloads))
    .sort((left, right) => compareVersions(right, left));
  return compatibleVersions[0] ?? null;
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
    desiredVersion,
    activeVersion,
    activePayloadPresent,
    desiredPayloadPresent,
  } = input;

  if (activeVersion === desiredVersion && activePayloadPresent) {
    return {
      action: "none",
      reason: `payload v${desiredVersion} is already active`,
    };
  }

  if (desiredPayloadPresent) {
    return {
      action: "switch",
      reason: `switching from payload v${activeVersion ?? "none"} to payload v${desiredVersion}`,
    };
  }

  return {
    action: "download",
    reason: `payload v${desiredVersion} is required, but it is not installed locally`,
  };
}
