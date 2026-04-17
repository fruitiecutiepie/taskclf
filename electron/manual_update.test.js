const test = require("node:test");
const assert = require("node:assert/strict");

const {
  combinedUpdateStateHasAnyUpdate,
  launcherResolutionSummary,
  manualUpdateDialogDetails,
} = require("./dist/manual_update.js");

function makeState(overrides = {}) {
  const launcherManifest = {
    launcher_version: "0.4.3",
    payload_index_url: "https://example.com/payload-index.json",
    compatible_payloads: {
      min_version_inclusive: "0.4.0",
      max_version_exclusive: "0.5.0",
    },
  };
  return {
    launcher: {
      currentVersion: "0.4.2",
      latestVersion: "0.4.3",
      latestTag: "launcher-v0.4.3",
      latestManifest: launcherManifest,
      platformData: {
        url: "https://example.com/taskclf.dmg",
        sha256: "abc123",
        filename: "taskclf.dmg",
      },
      updateAvailable: true,
    },
    payload: {
      launcherManifest,
      payloadIndex: {
        payloads: [
          { version: "0.4.8", manifest_url: "https://example.com/0.4.8.json" },
        ],
      },
      payloadManifest: {
        version: "0.4.8",
        platforms: {},
      },
      defaultVersion: "0.4.8",
      desiredVersion: "0.4.8",
      desiredSource: "default",
      selectedVersion: null,
      syncPlan: {
        action: "download",
        reason: "payload v0.4.8 is required, but it is not installed locally",
      },
    },
    activePayloadVersion: "0.4.7",
    selectedPayloadVersion: null,
    selectionIgnoredNote: null,
    compatiblePayloadVersions: ["0.4.8", "0.4.7"],
    ...overrides,
  };
}

test("combinedUpdateStateHasAnyUpdate is true when only launcher is actionable", () => {
  const base = makeState();
  const state = makeState({
    payload: {
      ...base.payload,
      syncPlan: { action: "none", reason: "payload v0.4.8 is already active" },
    },
  });
  assert.equal(combinedUpdateStateHasAnyUpdate(state), true);
});

test("combinedUpdateStateHasAnyUpdate is true when only core is actionable", () => {
  const base = makeState();
  const state = makeState({
    launcher: {
      ...base.launcher,
      platformData: null,
    },
  });
  assert.equal(combinedUpdateStateHasAnyUpdate(state), true);
});

test("combinedUpdateStateHasAnyUpdate is false when nothing is actionable", () => {
  const base = makeState();
  const state = makeState({
    launcher: {
      ...base.launcher,
      updateAvailable: false,
      platformData: null,
      latestVersion: "0.4.2",
      latestTag: "launcher-v0.4.2",
    },
    payload: {
      ...base.payload,
      syncPlan: { action: "none", reason: "payload v0.4.8 is already active" },
    },
  });
  assert.equal(combinedUpdateStateHasAnyUpdate(state), false);
});

test("launcherResolutionSummary explains missing platform installers", () => {
  const base = makeState();
  const state = makeState({
    launcher: {
      ...base.launcher,
      platformData: null,
    },
  });
  assert.match(
    launcherResolutionSummary(state.launcher),
    /no installer is published for this platform/i,
  );
});

test("manualUpdateDialogDetails includes launcher and core versions", () => {
  const detail = manualUpdateDialogDetails(makeState({
    selectionIgnoredNote: "Manual checks ignore the pinned core version.",
    selectedPayloadVersion: "0.4.6",
  }));
  assert.match(detail, /Launcher current: v0\.4\.2/);
  assert.match(detail, /Launcher latest: v0\.4\.3/);
  assert.match(detail, /Core active: 0\.4\.7/);
  assert.match(detail, /Core target: 0\.4\.8/);
  assert.match(detail, /Manual checks ignore the pinned core version\./);
});
