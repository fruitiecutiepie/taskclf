const test = require("node:test");
const assert = require("node:assert/strict");
const http = require("node:http");
const path = require("node:path");

const { nodeFetch } = require(path.join(__dirname, "dist", "node_http.js"));

async function withServer(handler, run) {
  const server = http.createServer(handler);
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  const address = server.address();
  assert.ok(address && typeof address === "object");
  const baseUrl = `http://127.0.0.1:${address.port}`;
  try {
    await run(baseUrl);
  } finally {
    await new Promise((resolve, reject) => {
      server.close((err) => {
        if (err) reject(err);
        else resolve();
      });
    });
  }
}

test("nodeFetch follows redirects and reads JSON responses", async () => {
  await withServer((req, res) => {
    if (req.url === "/start") {
      res.writeHead(302, { location: "/final" });
      res.end();
      return;
    }
    if (req.url === "/final") {
      res.writeHead(200, { "content-type": "application/json" });
      res.end(JSON.stringify({ ok: true }));
      return;
    }
    res.writeHead(404);
    res.end("missing");
  }, async (baseUrl) => {
    const response = await nodeFetch(`${baseUrl}/start`);
    assert.equal(response.status, 200);
    assert.deepEqual(await response.json(), { ok: true });
  });
});

test("nodeFetch preserves non-2xx HTTP responses", async () => {
  await withServer((_req, res) => {
    res.writeHead(404, { "content-type": "text/plain" });
    res.end("missing");
  }, async (baseUrl) => {
    const response = await nodeFetch(`${baseUrl}/missing`);
    assert.equal(response.ok, false);
    assert.equal(response.status, 404);
    assert.equal(await response.text(), "missing");
  });
});

test("nodeFetch aborts in-flight requests", async () => {
  await withServer((_req, res) => {
    setTimeout(() => {
      res.writeHead(200, { "content-type": "text/plain" });
      res.end("slow");
    }, 200);
  }, async (baseUrl) => {
    const controller = new AbortController();
    setTimeout(() => controller.abort(), 20);
    await assert.rejects(
      nodeFetch(`${baseUrl}/slow`, { signal: controller.signal }),
      (error) => error instanceof Error && error.name === "AbortError",
    );
  });
});
