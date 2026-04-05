const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");

const {
  APP_ICON_FILE,
  MACOS_TRAY_ICON_FILE,
  MACOS_TRAY_ICON_RETINA_FILE,
  getAppIconPath,
  getTrayAssetFiles,
  getTrayIconAsset,
} = require("./dist/tray_icon.js");

test("getTrayIconAsset uses a template image on macOS", () => {
  const baseDir = path.join("/tmp", "taskclf", "dist");
  assert.deepEqual(getTrayIconAsset(baseDir, "darwin"), {
    path: path.join("/tmp", "taskclf", "build", MACOS_TRAY_ICON_FILE),
    template: true,
  });
});

test("getTrayIconAsset falls back to the normal app icon on non-macOS platforms", () => {
  const baseDir = path.join("/tmp", "taskclf", "dist");
  assert.deepEqual(getTrayIconAsset(baseDir, "linux"), {
    path: path.join("/tmp", "taskclf", "build", APP_ICON_FILE),
    template: false,
  });
  assert.equal(getAppIconPath(baseDir), path.join("/tmp", "taskclf", "build", APP_ICON_FILE));
});

test("getTrayAssetFiles lists the packaged macOS tray assets", () => {
  assert.deepEqual(getTrayAssetFiles("darwin"), [
    `build/${MACOS_TRAY_ICON_FILE}`,
    `build/${MACOS_TRAY_ICON_RETINA_FILE}`,
  ]);
});

test("build step emits the tray template assets used by the packaged app", () => {
  for (const relativePath of getTrayAssetFiles("darwin")) {
    assert.equal(
      fs.existsSync(path.join(__dirname, relativePath)),
      true,
      `${relativePath} should exist after pnpm run build`,
    );
  }
});

test("electron packaging includes the tray template assets", () => {
  const pkg = JSON.parse(fs.readFileSync(path.join(__dirname, "package.json"), "utf8"));
  assert.equal(pkg.build.files.includes(`build/${MACOS_TRAY_ICON_FILE}`), true);
  assert.equal(pkg.build.files.includes(`build/${MACOS_TRAY_ICON_RETINA_FILE}`), true);
});
