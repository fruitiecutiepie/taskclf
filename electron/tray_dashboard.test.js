const test = require("node:test");
const assert = require("node:assert/strict");

const {
  dashboardWindowActionForTrayInteraction,
} = require("./dist/tray_dashboard.js");

test("tray icon clicks only show the dashboard", () => {
  assert.equal(
    dashboardWindowActionForTrayInteraction("icon-click"),
    "show",
  );
});

test("tray menu toggle keeps hide/show behavior", () => {
  assert.equal(
    dashboardWindowActionForTrayInteraction("menu-toggle"),
    "toggle",
  );
});
