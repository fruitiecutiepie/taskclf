import http, { STATUS_CODES, type IncomingHttpHeaders } from "node:http";
import https from "node:https";
import { Readable } from "node:stream";

export interface NodeFetchInit extends RequestInit {
  maxRedirects?: number;
}

const REDIRECT_STATUS_CODES = new Set([301, 302, 303, 307, 308]);
const NULL_BODY_STATUS_CODES = new Set([204, 205, 304]);

function abortError(reason?: unknown): Error {
  if (reason instanceof Error) {
    return reason;
  }
  return new DOMException(
    reason === undefined ? "The operation was aborted." : String(reason),
    "AbortError",
  );
}

function requestTransport(url: URL): typeof http | typeof https {
  if (url.protocol === "http:") {
    return http;
  }
  if (url.protocol === "https:") {
    return https;
  }
  throw new Error(`Unsupported protocol for nodeFetch: ${url.protocol}`);
}

function responseHeaders(headers: IncomingHttpHeaders): Headers {
  const normalized = new Headers();
  for (const [key, value] of Object.entries(headers)) {
    if (value === undefined) {
      continue;
    }
    normalized.set(key, Array.isArray(value) ? value.join(", ") : value);
  }
  return normalized;
}

function redirectMethod(status: number, method: string): string {
  if (status === 303) {
    return "GET";
  }
  if ((status === 301 || status === 302) && method.toUpperCase() === "POST") {
    return "GET";
  }
  return method;
}

async function bodyBufferFor(request: Request): Promise<Buffer | null> {
  if (request.body === null) {
    return null;
  }
  return Buffer.from(await request.arrayBuffer());
}

function requestHeaders(headers: Headers): Record<string, string> {
  const values = Object.fromEntries(headers.entries());
  if (!("accept-encoding" in values)) {
    values["accept-encoding"] = "identity";
  }
  return values;
}

async function nodeFetchRequest(
  request: Request,
  bodyBuffer: Buffer | null,
  remainingRedirects: number,
): Promise<Response> {
  if (request.signal.aborted) {
    throw abortError((request.signal as AbortSignal & { reason?: unknown }).reason);
  }

  const url = new URL(request.url);
  const transport = requestTransport(url);

  return await new Promise<Response>((resolve, reject) => {
    let settled = false;

    const fail = (error: unknown) => {
      if (settled) {
        return;
      }
      settled = true;
      cleanup();
      reject(error instanceof Error ? error : new Error(String(error)));
    };

    const succeed = (response: Response | Promise<Response>) => {
      if (settled) {
        return;
      }
      settled = true;
      cleanup();
      resolve(response);
    };

    const req = transport.request(
      {
        protocol: url.protocol,
        hostname: url.hostname,
        port: url.port === "" ? undefined : url.port,
        path: `${url.pathname}${url.search}`,
        method: request.method,
        headers: requestHeaders(new Headers(request.headers)),
      },
      (res) => {
        const status = res.statusCode ?? 0;
        const location = res.headers.location;

        if (REDIRECT_STATUS_CODES.has(status) && location) {
          if (request.redirect === "error") {
            res.resume();
            fail(new Error(`Redirect encountered while fetching ${request.url}`));
            return;
          }
          if (request.redirect !== "manual") {
            if (remainingRedirects <= 0) {
              res.resume();
              fail(new Error(`Too many redirects while fetching ${request.url}`));
              return;
            }

            const nextUrl = new URL(location, request.url);
            const nextMethod = redirectMethod(status, request.method);
            const nextHeaders = new Headers(request.headers);
            const nextBody = nextMethod === request.method ? bodyBuffer : null;
            if (nextBody === null) {
              nextHeaders.delete("content-length");
              if (nextMethod === "GET") {
                nextHeaders.delete("content-type");
              }
            }
            res.resume();
            succeed(nodeFetchRequest(
              new Request(nextUrl, {
                method: nextMethod,
                headers: nextHeaders,
                body: nextBody === null ? undefined : new Uint8Array(nextBody),
                signal: request.signal,
                redirect: request.redirect,
              }),
              nextBody,
              remainingRedirects - 1,
            ));
            return;
          }
        }

        if (status < 200 || status > 599) {
          res.resume();
          fail(new Error(`Received unsupported HTTP status ${status} while fetching ${request.url}`));
          return;
        }

        const body = NULL_BODY_STATUS_CODES.has(status) || request.method === "HEAD"
          ? null
          : (Readable.toWeb(res as unknown as Readable) as ReadableStream);
        succeed(new Response(body, {
          status,
          statusText: res.statusMessage || STATUS_CODES[status] || "",
          headers: responseHeaders(res.headers),
        }));
      },
    );

    req.on("error", fail);

    const onAbort = () => {
      const error = abortError((request.signal as AbortSignal & { reason?: unknown }).reason);
      req.destroy(error);
      fail(error);
    };

    const cleanup = () => {
      request.signal.removeEventListener("abort", onAbort);
    };

    request.signal.addEventListener("abort", onAbort, { once: true });

    if (bodyBuffer !== null && bodyBuffer.byteLength > 0) {
      req.end(bodyBuffer);
      return;
    }
    req.end();
  });
}

export async function nodeFetch(
  input: string | URL | Request,
  init?: NodeFetchInit,
): Promise<Response> {
  const request = input instanceof Request && init === undefined
    ? input
    : new Request(input, init);
  const bodyBuffer = await bodyBufferFor(request);
  return nodeFetchRequest(request, bodyBuffer, init?.maxRedirects ?? 10);
}
