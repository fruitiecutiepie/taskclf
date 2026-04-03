const test = require("node:test");
const assert = require("node:assert/strict");

const {
  manifestUrlForLauncherVersion,
  resolvePayloadSyncPlan,
} = require("./dist/update_policy.js");

test("manifestUrlForLauncherVersion pins the packaged app to its launcher tag", () => {
  assert.equal(
    manifestUrlForLauncherVersion("0.3.19"),
    "https://github.com/fruitiecutiepie/taskclf/releases/download/launcher-v0.3.19/manifest.json",
  );
});

test("manifestUrlForLauncherVersion honors TASKCLF_MANIFEST_URL override", () => {
  assert.equal(
    manifestUrlForLauncherVersion("0.3.19", "https://example.com/manifest.json"),
    "https://example.com/manifest.json",
  );
});

test("resolvePayloadSyncPlan returns none when the launcher-matched payload is already active", () => {
  assert.deepEqual(
    resolvePayloadSyncPlan({
      launcherVersion: "0.3.19",
      activeVersion: "0.3.19",
      activePayloadPresent: true,
      launcherPayloadPresent: true,
    }),
    {
      action: "none",
      reason: "payload v0.3.19 already matches the launcher",
    },
  );
});

test("resolvePayloadSyncPlan switches back to a cached launcher-matched payload", () => {
  assert.deepEqual(
    resolvePayloadSyncPlan({
      launcherVersion: "0.3.19",
      activeVersion: "0.4.4",
      activePayloadPresent: true,
      launcherPayloadPresent: true,
    }),
    {
      action: "switch",
      reason: "switching from payload v0.4.4 to launcher-matched payload v0.3.19",
    },
  );
});

test("resolvePayloadSyncPlan downloads when the launcher-matched payload is missing", () => {
  assert.deepEqual(
    resolvePayloadSyncPlan({
      launcherVersion: "0.3.19",
      activeVersion: "0.4.4",
      activePayloadPresent: true,
      launcherPayloadPresent: false,
    }),
    {
      action: "download",
      reason: "launcher v0.3.19 requires payload v0.3.19, but the payload is not installed locally",
    },
  );
});
