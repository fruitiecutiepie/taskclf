const test = require("node:test");
const assert = require("node:assert/strict");

const {
  orderedCompatiblePayloadVersions,
  installableOnlyPayloadVersions,
  payloadChooserOffersMultipleVersions,
} = require("./dist/payload_choice.js");

test("orderedCompatiblePayloadVersions lists newest compatible first", () => {
  const compatiblePayloads = {
    min_version_inclusive: "0.4.0",
    max_version_exclusive: "0.5.0",
  };
  const payloads = [
    { version: "0.4.1" },
    { version: "0.5.0" },
    { version: "0.4.10" },
  ];
  assert.deepEqual(
    orderedCompatiblePayloadVersions(payloads, compatiblePayloads),
    ["0.4.10", "0.4.1"],
  );
});

test("installableOnlyPayloadVersions excludes installed", () => {
  assert.deepEqual(
    installableOnlyPayloadVersions(["0.4.4", "0.4.10"], ["0.4.10"]),
    ["0.4.4"],
  );
});

test("payloadChooserOffersMultipleVersions is true only when more than one choice", () => {
  assert.equal(payloadChooserOffersMultipleVersions(["0.4.4", "0.4.10"]), true);
  assert.equal(payloadChooserOffersMultipleVersions(["0.4.4"]), false);
});
