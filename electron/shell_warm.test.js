const test = require("node:test");
const assert = require("node:assert/strict");

const { shellLoadingDataUrl } = require("./dist/shell_warm.js");

test("shellLoadingDataUrl returns a data URL with loading placeholder", () => {
  const url = shellLoadingDataUrl();
  assert.ok(url.startsWith("data:text/html;charset=utf-8,"));
  assert.ok(decodeURIComponent(url.slice("data:text/html;charset=utf-8,".length)).includes("Loading"));
});
