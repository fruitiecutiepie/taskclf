/**
 * Pure helpers for compatible payload version lists (tray menu, install/update choosers).
 * Keeps ordering and filtering aligned with {@link ./update_policy}.
 */
import type { CompatiblePayloadRange } from "./update_policy";
import { compareVersions, isPayloadVersionCompatible } from "./update_policy";

export interface PayloadVersionEntry {
  version: string;
}

/**
 * Compatible payload versions from the index, newest first (same as tray Payload submenu).
 */
export function orderedCompatiblePayloadVersions(
  payloads: PayloadVersionEntry[],
  compatiblePayloads: CompatiblePayloadRange,
): string[] {
  return payloads
    .map((entry) => entry.version)
    .filter((version) => isPayloadVersionCompatible(version, compatiblePayloads))
    .sort((left, right) => compareVersions(right, left));
}

/**
 * Versions in the index that are compatible but not yet installed locally.
 */
export function installableOnlyPayloadVersions(
  compatibleOrderedDescending: string[],
  installedVersions: string[],
): string[] {
  return compatibleOrderedDescending.filter(
    (version) => !installedVersions.includes(version),
  );
}

/** True when the optional install/update chooser should be offered (more than one choice). */
export function payloadChooserOffersMultipleVersions(
  compatibleOrderedDescending: string[],
): boolean {
  return compatibleOrderedDescending.length > 1;
}
