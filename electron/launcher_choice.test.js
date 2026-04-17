const test = require("node:test");
const assert = require("node:assert/strict");

const {
  launcherVersionFromTag,
  selectLatestLauncherReleaseTag,
  launcherUpdateAvailable,
} = require("./dist/launcher_choice.js");

test("launcherVersionFromTag parses launcher tags only", () => {
  assert.equal(launcherVersionFromTag("launcher-v0.4.2"), "0.4.2");
  assert.equal(launcherVersionFromTag("v0.4.2"), null);
  assert.equal(launcherVersionFromTag("launcher-vlatest"), null);
});

test("selectLatestLauncherReleaseTag ignores non-launcher, draft, and prerelease tags", () => {
  assert.equal(
    selectLatestLauncherReleaseTag([
      { tag_name: "v0.4.9" },
      { tag_name: "launcher-v0.4.1", draft: true },
      { tag_name: "launcher-v0.4.3", prerelease: true },
      { tag_name: "launcher-v0.4.2" },
      { tag_name: "launcher-v0.4.10" },
    ]),
    "launcher-v0.4.10",
  );
});

test("launcherUpdateAvailable detects newer launcher versions", () => {
  assert.equal(launcherUpdateAvailable("0.4.2", "0.4.3"), true);
  assert.equal(launcherUpdateAvailable("0.4.2", "0.4.2"), false);
  assert.equal(launcherUpdateAvailable("0.4.3", "0.4.2"), false);
});
