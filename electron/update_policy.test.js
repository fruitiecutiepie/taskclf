const test = require("node:test");
const assert = require("node:assert/strict");

const {
  compareVersions,
  isPayloadVersionCompatible,
  manifestUrlForLauncherVersion,
  selectLatestCompatiblePayloadVersion,
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

test("compareVersions orders semantic versions numerically", () => {
  assert.equal(compareVersions("0.4.10", "0.4.4"), 1);
  assert.equal(compareVersions("0.4.4", "0.4.10"), -1);
  assert.equal(compareVersions("0.4.4", "0.4.4"), 0);
});

test("isPayloadVersionCompatible enforces inclusive min and exclusive max", () => {
  const compatiblePayloads = {
    min_version_inclusive: "0.4.0",
    max_version_exclusive: "0.5.0",
  };
  assert.equal(isPayloadVersionCompatible("0.4.0", compatiblePayloads), true);
  assert.equal(isPayloadVersionCompatible("0.4.4", compatiblePayloads), true);
  assert.equal(isPayloadVersionCompatible("0.5.0", compatiblePayloads), false);
  assert.equal(isPayloadVersionCompatible("0.3.99", compatiblePayloads), false);
});

test("selectLatestCompatiblePayloadVersion picks newest allowed payload", () => {
  assert.equal(
    selectLatestCompatiblePayloadVersion({
      compatiblePayloads: {
        min_version_inclusive: "0.4.0",
        max_version_exclusive: "0.5.0",
      },
      availableVersions: ["0.4.1", "0.5.1", "0.4.10", "0.3.9", "0.4.4"],
    }),
    "0.4.10",
  );
});

test("selectLatestCompatiblePayloadVersion returns null when none are compatible", () => {
  assert.equal(
    selectLatestCompatiblePayloadVersion({
      compatiblePayloads: {
        min_version_inclusive: "0.4.0",
        max_version_exclusive: "0.5.0",
      },
      availableVersions: ["0.3.9", "0.5.0"],
    }),
    null,
  );
});

test("resolvePayloadSyncPlan returns none when the desired payload is already active", () => {
  assert.deepEqual(
    resolvePayloadSyncPlan({
      desiredVersion: "0.4.4",
      activeVersion: "0.4.4",
      activePayloadPresent: true,
      desiredPayloadPresent: true,
    }),
    {
      action: "none",
      reason: "payload v0.4.4 is already active",
    },
  );
});

test("resolvePayloadSyncPlan switches to a cached desired payload", () => {
  assert.deepEqual(
    resolvePayloadSyncPlan({
      desiredVersion: "0.4.4",
      activeVersion: "0.4.1",
      activePayloadPresent: true,
      desiredPayloadPresent: true,
    }),
    {
      action: "switch",
      reason: "switching from payload v0.4.1 to payload v0.4.4",
    },
  );
});

test("resolvePayloadSyncPlan downloads when the desired payload is missing", () => {
  assert.deepEqual(
    resolvePayloadSyncPlan({
      desiredVersion: "0.4.4",
      activeVersion: "0.4.4",
      activePayloadPresent: false,
      desiredPayloadPresent: false,
    }),
    {
      action: "download",
      reason: "payload v0.4.4 is required, but it is not installed locally",
    },
  );
});
