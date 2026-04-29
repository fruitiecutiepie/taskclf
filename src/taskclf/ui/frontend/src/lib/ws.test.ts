import { render, waitFor } from "@solidjs/testing-library";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { suggestion_banner_ttl_ms_from_seconds, ws_store_new } from "./ws";

describe("suggestion_banner_ttl_ms_from_seconds", () => {
  it("returns null when disabled or invalid", () => {
    expect(suggestion_banner_ttl_ms_from_seconds(0)).toBeNull();
    expect(suggestion_banner_ttl_ms_from_seconds(-1)).toBeNull();
    expect(suggestion_banner_ttl_ms_from_seconds(Number.NaN)).toBeNull();
    expect(suggestion_banner_ttl_ms_from_seconds(Number.POSITIVE_INFINITY)).toBeNull();
  });

  it("returns milliseconds for positive seconds", () => {
    expect(suggestion_banner_ttl_ms_from_seconds(1)).toBe(1000);
    expect(suggestion_banner_ttl_ms_from_seconds(600)).toBe(600_000);
    expect(suggestion_banner_ttl_ms_from_seconds(1.7)).toBe(1000);
  });
});

class MockWebSocket {
  static readonly CONNECTING = 0;
  static readonly OPEN = 1;
  static readonly CLOSING = 2;
  static readonly CLOSED = 3;
  static instances: MockWebSocket[] = [];

  readonly url: string;
  readyState = MockWebSocket.CONNECTING;
  onopen: ((event: Event) => void) | null = null;
  onmessage: ((event: MessageEvent) => void) | null = null;
  onclose: ((event: CloseEvent) => void) | null = null;
  onerror: ((event: Event) => void) | null = null;

  constructor(url: string) {
    this.url = url;
    MockWebSocket.instances.push(this);
  }

  emit_open() {
    this.readyState = MockWebSocket.OPEN;
    this.onopen?.(new Event("open"));
  }

  emit_message(data: unknown) {
    this.onmessage?.({ data: JSON.stringify(data) } as MessageEvent);
  }

  close() {
    this.readyState = MockWebSocket.CLOSED;
  }
}

function response_json(data: unknown): Response {
  return {
    ok: true,
    json: async () => data,
  } as Response;
}

describe("ws_store_new badge display override", () => {
  const unmounts: Array<() => void> = [];

  beforeEach(() => {
    MockWebSocket.instances = [];
    vi.stubGlobal("WebSocket", MockWebSocket as unknown as typeof WebSocket);
    vi.stubGlobal(
      "fetch",
      vi.fn((input: RequestInfo | URL) => {
        const url = String(input);
        if (url === "/api/config/user") {
          return Promise.resolve(
            response_json({
              user_id: "tester",
              username: "tester",
              suggestion_banner_ttl_seconds: 0,
            }),
          );
        }
        if (url === "/api/ws/snapshot") {
          return Promise.resolve(response_json({}));
        }
        return Promise.reject(new Error(`Unexpected fetch in ws.test.ts: ${url}`));
      }),
    );
  });

  afterEach(() => {
    for (const unmount of unmounts.splice(0)) {
      unmount();
    }
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
  });

  function ws_store_mount() {
    let store!: ReturnType<typeof ws_store_new>;
    const mounted = render(() => {
      store = ws_store_new();
      return null;
    });
    unmounts.push(mounted.unmount);
    return store;
  }

  async function ws_store_connect() {
    const store = ws_store_mount();
    await waitFor(() => {
      expect(MockWebSocket.instances).toHaveLength(1);
    });
    const socket = MockWebSocket.instances[0];
    socket.emit_open();
    await waitFor(() => {
      expect(store.connection_status()).toBe("connected");
    });
    return { store, socket };
  }

  it("restores the pre-suggestion badge display when the suggestion is skipped", async () => {
    const { store, socket } = await ws_store_connect();

    socket.emit_message({
      type: "live_status",
      label: "Write",
      text: "Now: Write",
      ts: "2026-04-05T10:00:00Z",
    });
    await waitFor(() => {
      expect(store.live_status()?.label).toBe("Write");
    });

    socket.emit_message({
      type: "suggest_label",
      reason: "app_switch",
      old_label: "Write",
      suggested: "Review",
      confidence: 0.93,
      block_start: "2026-04-05T09:30:00Z",
      block_end: "2026-04-05T10:00:00Z",
    });
    await waitFor(() => {
      expect(store.badge_display_override()).toEqual({
        enabled: true,
        label: "Review",
      });
    });

    socket.emit_message({
      type: "suggestion_cleared",
      reason: "skipped",
    });
    await waitFor(() => {
      expect(store.active_suggestion()).toBeNull();
    });
    expect(store.badge_display_override()).toEqual({
      enabled: true,
      label: "Write",
    });

    socket.emit_message({
      type: "prediction",
      label: "Build",
      mapped_label: "Build",
      confidence: 0.88,
      ts: "2026-04-05T10:01:00Z",
    });
    await waitFor(() => {
      expect(store.badge_display_override()).toEqual({
        enabled: false,
        label: null,
      });
    });
  });

  it("keeps the accepted suggestion until a fresher explicit badge signal arrives", async () => {
    const { store, socket } = await ws_store_connect();

    socket.emit_message({
      type: "live_status",
      label: "Write",
      text: "Now: Write",
      ts: "2026-04-05T10:00:00Z",
    });
    await waitFor(() => {
      expect(store.live_status()?.label).toBe("Write");
    });

    socket.emit_message({
      type: "suggest_label",
      reason: "app_switch",
      old_label: "Write",
      suggested: "Review",
      confidence: 0.93,
      block_start: "2026-04-05T09:30:00Z",
      block_end: "2026-04-05T10:00:00Z",
    });
    await waitFor(() => {
      expect(store.badge_display_override()).toEqual({
        enabled: true,
        label: "Review",
      });
    });

    socket.emit_message({
      type: "suggestion_cleared",
      reason: "label_saved",
    });
    await waitFor(() => {
      expect(store.active_suggestion()).toBeNull();
    });
    expect(store.badge_display_override()).toEqual({
      enabled: true,
      label: "Review",
    });

    socket.emit_message({
      type: "live_status",
      label: "Build",
      text: "Now: Build",
      ts: "2026-04-05T10:02:00Z",
    });
    await waitFor(() => {
      expect(store.badge_display_override()).toEqual({
        enabled: false,
        label: null,
      });
    });
  });

  it("queues distinct suggestions and clears only the targeted item", async () => {
    const { store, socket } = await ws_store_connect();

    const first = {
      type: "suggest_label" as const,
      suggestion_id: "2026-04-05T09:30:00Z|2026-04-05T10:00:00Z",
      reason: "app_switch",
      old_label: "Write",
      suggested: "Review",
      confidence: 0.93,
      block_start: "2026-04-05T09:30:00Z",
      block_end: "2026-04-05T10:00:00Z",
    };
    const second = {
      type: "suggest_label" as const,
      suggestion_id: "2026-04-05T10:00:00Z|2026-04-05T10:30:00Z",
      reason: "app_switch",
      old_label: "Review",
      suggested: "Build",
      confidence: 0.88,
      block_start: "2026-04-05T10:00:00Z",
      block_end: "2026-04-05T10:30:00Z",
    };

    socket.emit_message(first);
    socket.emit_message(second);

    await waitFor(() => {
      expect(store.pending_suggestions()).toHaveLength(2);
    });
    expect(store.active_suggestion()?.suggested).toBe("Review");
    expect(store.badge_display_override()).toEqual({
      enabled: true,
      label: "Review",
    });

    store.suggestion_select(second);
    expect(store.active_suggestion()?.suggested).toBe("Build");
    expect(store.badge_display_override()).toEqual({
      enabled: true,
      label: "Build",
    });

    socket.emit_message({
      type: "suggestion_cleared",
      reason: "skipped",
      suggestion_id: second.suggestion_id,
    });

    await waitFor(() => {
      expect(store.pending_suggestions()).toHaveLength(1);
    });
    expect(store.active_suggestion()?.suggested).toBe("Review");
    expect(store.badge_display_override()).toEqual({
      enabled: true,
      label: "Review",
    });
  });
});
