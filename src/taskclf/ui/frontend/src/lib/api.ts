const BASE = "/api";

export interface LabelResponse {
  start_ts: string;
  end_ts: string;
  label: string;
  provenance: string;
  user_id: string | null;
  confidence: number | null;
}

export interface QueueItem {
  request_id: string;
  user_id: string;
  bucket_start_ts: string;
  bucket_end_ts: string;
  reason: string;
  confidence: number | null;
  predicted_label: string | null;
  status: string;
}

export interface FeatureSummary {
  top_apps: { app_id: string; buckets: number }[];
  mean_keys_per_min: number | null;
  mean_clicks_per_min: number | null;
  mean_scroll_per_min: number | null;
  total_buckets: number;
  session_count: number;
}

export interface AWLiveEntry {
  app: string;
  events: number;
}

async function json<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init);
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`${res.status}: ${text}`);
  }
  return res.json();
}

export async function fetchLabels(limit = 50): Promise<LabelResponse[]> {
  return json(`${BASE}/labels?limit=${limit}`);
}

export async function createLabel(body: {
  start_ts: string;
  end_ts: string;
  label: string;
  user_id?: string;
  confidence?: number;
  extend_previous?: boolean;
}): Promise<LabelResponse> {
  return json(`${BASE}/labels`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

export async function fetchQueue(limit = 20): Promise<QueueItem[]> {
  return json(`${BASE}/queue?limit=${limit}`);
}

export async function markQueueDone(
  requestId: string,
  status: "labeled" | "skipped" = "labeled"
): Promise<{ status: string }> {
  return json(`${BASE}/queue/${requestId}/done`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ status }),
  });
}

export async function fetchFeatureSummary(
  start: string,
  end: string
): Promise<FeatureSummary> {
  return json(
    `${BASE}/features/summary?start=${encodeURIComponent(start)}&end=${encodeURIComponent(end)}`
  );
}

export async function fetchAWLive(
  start: string,
  end: string
): Promise<AWLiveEntry[]> {
  return json(
    `${BASE}/aw/live?start=${encodeURIComponent(start)}&end=${encodeURIComponent(end)}`
  );
}

export async function fetchCoreLabels(): Promise<string[]> {
  return json(`${BASE}/config/labels`);
}
