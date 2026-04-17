import { compareVersions } from "./update_policy";

export interface LauncherReleaseEntry {
  tag_name: string;
  draft?: boolean;
  prerelease?: boolean;
}

export function launcherVersionFromTag(tag: string): string | null {
  const prefix = "launcher-v";
  if (!tag.startsWith(prefix)) {
    return null;
  }
  const version = tag.slice(prefix.length);
  return /^\d+\.\d+\.\d+$/.test(version) ? version : null;
}

export function selectLatestLauncherReleaseTag(
  releases: LauncherReleaseEntry[],
): string | null {
  const matching = releases
    .filter((release) => !release.draft && !release.prerelease)
    .map((release) => ({
      tag: release.tag_name,
      version: launcherVersionFromTag(release.tag_name),
    }))
    .filter((entry): entry is { tag: string; version: string } => entry.version !== null)
    .sort((left, right) => compareVersions(right.version, left.version));
  return matching[0]?.tag ?? null;
}

export function launcherUpdateAvailable(
  currentVersion: string,
  latestVersion: string,
): boolean {
  return compareVersions(latestVersion, currentVersion) > 0;
}
