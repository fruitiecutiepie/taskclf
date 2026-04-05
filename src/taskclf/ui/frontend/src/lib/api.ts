const BASE = "/api";

export type LabelResponse = {
  start_ts: string;
  end_ts: string;
  label: string;
  provenance: string;
  user_id: string | null;
  confidence: number | null;
  extend_forward: boolean;
};

export type QueueItem = {
  request_id: string;
  user_id: string;
  bucket_start_ts: string;
  bucket_end_ts: string;
  reason: string;
  confidence: number | null;
  predicted_label: string | null;
  status: string;
};

export type FeatureSummary = {
  top_apps: { app_id: string; buckets: number }[];
  mean_keys_per_min: number | null;
  mean_clicks_per_min: number | null;
  mean_scroll_per_min: number | null;
  total_buckets: number;
  session_count: number;
};

export type AWLiveEntry = {
  app: string;
  events: number;
};

async function api_json<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init);
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`${res.status}: ${text}`);
  }
  return res.json();
}

export async function labels_list(limit = 50): Promise<LabelResponse[]> {
  return api_json(`${BASE}/labels?limit=${limit}`);
}

export async function labels_list_by_date(date: string): Promise<LabelResponse[]> {
  const day_start = new Date(`${date}T00:00:00`);
  const day_end = new Date(`${date}T23:59:59.999`);
  const rs = encodeURIComponent(day_start.toISOString());
  const re = encodeURIComponent(day_end.toISOString());
  return api_json(`${BASE}/labels?limit=500&range_start=${rs}&range_end=${re}`);
}

export async function label_create(body: {
  start_ts: string;
  end_ts: string;
  label: string;
  user_id?: string;
  confidence?: number;
  extend_forward?: boolean;
  overwrite?: boolean;
  allow_overlap?: boolean;
}): Promise<LabelResponse> {
  return api_json(`${BASE}/labels`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

export async function queue_list(limit = 20): Promise<QueueItem[]> {
  return api_json(`${BASE}/queue?limit=${limit}`);
}

export async function queue_done_mark(
  request_id: string,
  status: "labeled" | "skipped" = "labeled",
): Promise<{ status: string }> {
  return api_json(`${BASE}/queue/${request_id}/done`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ status }),
  });
}

export async function feature_summary_get(
  start: string,
  end: string,
): Promise<FeatureSummary> {
  return api_json(
    `${BASE}/features/summary?start=${encodeURIComponent(start)}&end=${encodeURIComponent(end)}`,
  );
}

export async function aw_live_list(start: string, end: string): Promise<AWLiveEntry[]> {
  return api_json(
    `${BASE}/aw/live?start=${encodeURIComponent(start)}&end=${encodeURIComponent(end)}`,
  );
}

export async function label_update(body: {
  start_ts: string;
  end_ts: string;
  label: string;
  new_start_ts?: string;
  new_end_ts?: string;
  extend_forward?: boolean;
}): Promise<LabelResponse> {
  return api_json(`${BASE}/labels`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

export async function label_delete(body: {
  start_ts: string;
  end_ts: string;
}): Promise<{ status: string }> {
  return api_json(`${BASE}/labels`, {
    method: "DELETE",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

export async function core_labels_list(): Promise<string[]> {
  return api_json(`${BASE}/config/labels`);
}

export async function notification_accept(body: {
  block_start: string;
  block_end: string;
  label: string;
}): Promise<LabelResponse> {
  return api_json(`${BASE}/notification/accept`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

export async function notification_skip(): Promise<{ status: string }> {
  return api_json(`${BASE}/notification/skip`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
  });
}

export type UserConfig = {
  user_id: string;
  username: string;
};

export async function user_config_get(): Promise<UserConfig> {
  return api_json(`${BASE}/config/user`);
}

export async function user_config_update(patch: {
  username?: string;
}): Promise<UserConfig> {
  return api_json(`${BASE}/config/user`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(patch),
  });
}

// -- Training ----------------------------------------------------------------

export type TrainStatus = {
  job_id: string | null;
  status: "idle" | "running" | "complete" | "failed";
  step: string | null;
  progress_pct: number | null;
  message: string | null;
  error: string | null;
  metrics: Record<string, unknown> | null;
  model_dir: string | null;
  started_at: string | null;
  finished_at: string | null;
};

export type ModelBundle = {
  model_id: string;
  path: string;
  valid: boolean;
  invalid_reason: string | null;
  macro_f1: number | null;
  weighted_f1: number | null;
  created_at: string | null;
};

export type DataCheck = {
  date_from: string;
  date_to: string;
  dates_with_features: string[];
  dates_missing_features: string[];
  total_feature_rows: number;
  label_span_count: number;
  trainable_rows: number;
  trainable_labels: string[];
  dates_built: string[];
  build_errors: string[];
};

export async function training_start(params: {
  date_from: string;
  date_to: string;
  num_boost_round?: number;
  class_weight?: "balanced" | "none";
  synthetic?: boolean;
}): Promise<TrainStatus> {
  return api_json(`${BASE}/train/start`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
  });
}

export async function training_features_build(params: {
  date_from: string;
  date_to: string;
}): Promise<TrainStatus> {
  return api_json(`${BASE}/train/build-features`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
  });
}

export async function training_status_get(): Promise<TrainStatus> {
  return api_json(`${BASE}/train/status`);
}

export async function training_cancel(): Promise<TrainStatus> {
  return api_json(`${BASE}/train/cancel`, { method: "POST" });
}

export async function models_list(): Promise<ModelBundle[]> {
  return api_json(`${BASE}/train/models`);
}

export async function training_data_check(
  date_from: string,
  date_to: string,
): Promise<DataCheck> {
  return api_json(
    `${BASE}/train/data-check?date_from=${encodeURIComponent(date_from)}&date_to=${encodeURIComponent(date_to)}`,
  );
}
