const test = require("node:test");
const assert = require("node:assert/strict");
const path = require("node:path");

const pc = require(path.join(__dirname, "dist", "port_conflict.js"));

test("classifyListenerCommandLine marks packaged backend as taskclf", () => {
  const cmd =
    "/Users/x/Library/Application Support/taskclf-electron/versions/v0.4.6/backend/entry tray --browser --no-tray --port 8741";
  assert.equal(pc.classifyListenerCommandLine(cmd), "taskclf");
});

test("classifyListenerCommandLine marks dev python sidecar as taskclf", () => {
  const cmd =
    "/usr/bin/python3 -m taskclf.cli.main tray --browser --no-tray --no-open-browser --port 8741";
  assert.equal(pc.classifyListenerCommandLine(cmd), "taskclf");
});

test("classifyListenerCommandLine marks unrelated server as non-taskclf", () => {
  assert.equal(pc.classifyListenerCommandLine("/usr/sbin/nginx"), "non-taskclf");
});

test("classifyListenerCommandLine empty is unknown", () => {
  assert.equal(pc.classifyListenerCommandLine(""), "unknown");
});

test("parsePidFromSsOutput extracts pid", () => {
  const sample =
    "LISTEN 0 128 127.0.0.1:8741 users:((\"entry\",pid=82904,fd=10))";
  assert.equal(pc.parsePidFromSsOutput(sample), 82904);
  assert.equal(pc.parsePidFromSsOutput("no pid here"), null);
});

test("getPortListenerInfo returns null when port is free (lsof empty)", () => {
  const mockSpawn = (cmd, args) => {
    if (cmd === "lsof" && args.includes("-t")) {
      return { status: 1, stdout: "", stderr: "" };
    }
    throw new Error(`unexpected: ${cmd} ${args.join(" ")}`);
  };
  assert.equal(pc.getPortListenerInfo(8741, "darwin", mockSpawn), null);
});

test("getPortListenerInfo classifies taskclf listener", () => {
  const backendCmd =
    "/home/x/taskclf-electron/versions/v1/backend/entry tray --browser --no-tray --port 8741";
  const mockSpawn = (cmd, args) => {
    if (cmd === "lsof" && args.includes("-t")) {
      return { status: 0, stdout: "12345\n", stderr: "" };
    }
    if (cmd === "ps") {
      assert.equal(args[1], "12345");
      return { status: 0, stdout: `${backendCmd}\n`, stderr: "" };
    }
    throw new Error(`unexpected: ${cmd} ${args.join(" ")}`);
  };
  const info = pc.getPortListenerInfo(8741, "darwin", mockSpawn);
  assert.ok(info);
  assert.equal(info.pid, 12345);
  assert.equal(info.kind, "taskclf");
  assert.ok(info.commandLine.includes("backend/entry"));
});

test("getPortListenerInfo uses ss fallback on linux when lsof has no pid", () => {
  const ssOut =
    'LISTEN 0 128 127.0.0.1:8741 users:(("nginx",pid=99,fd=3))';
  const mockSpawn = (cmd, args) => {
    if (cmd === "lsof" && args.includes("-t")) {
      return { status: 1, stdout: "", stderr: "" };
    }
    if (cmd === "ss") {
      assert.ok(args.some((a) => a.includes("8741")));
      return { status: 0, stdout: ssOut, stderr: "" };
    }
    if (cmd === "ps") {
      return { status: 0, stdout: "/usr/sbin/nginx\n", stderr: "" };
    }
    throw new Error(`unexpected: ${cmd}`);
  };
  const info = pc.getPortListenerInfo(8741, "linux", mockSpawn);
  assert.ok(info);
  assert.equal(info.pid, 99);
  assert.equal(info.kind, "non-taskclf");
});

test("killPidAndWaitForPortFree stops polling when port clears", async () => {
  let portBusy = true;
  const mockSpawn = (cmd, args) => {
    if (cmd === "lsof" && args.includes("-t")) {
      if (portBusy) {
        return { status: 0, stdout: "1\n", stderr: "" };
      }
      return { status: 1, stdout: "", stderr: "" };
    }
    if (cmd === "kill") {
      portBusy = false;
      return { status: 0, stdout: "", stderr: "" };
    }
    throw new Error(`unexpected: ${cmd}`);
  };
  const ok = await pc.killPidAndWaitForPortFree(8741, 1, "darwin", mockSpawn, 5000);
  assert.equal(ok, true);
});

test("killPidAndWaitForPortFree returns false if port stays busy", async () => {
  const mockSpawn = (cmd, args) => {
    if (cmd === "lsof" && args.includes("-t")) {
      return { status: 0, stdout: "1\n", stderr: "" };
    }
    if (cmd === "ps") {
      return { status: 0, stdout: "/usr/sbin/nginx\n", stderr: "" };
    }
    if (cmd === "kill") {
      return { status: 0, stdout: "", stderr: "" };
    }
    throw new Error(`unexpected: ${cmd}`);
  };
  const ok = await pc.killPidAndWaitForPortFree(8741, 1, "darwin", mockSpawn, 80);
  assert.equal(ok, false);
});
