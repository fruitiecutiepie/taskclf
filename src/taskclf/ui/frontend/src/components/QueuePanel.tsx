import {
  type Component,
  createResource,
  For,
  Show,
} from "solid-js";
import { fetchQueue, markQueueDone } from "../lib/api";

export const QueuePanel: Component = () => {
  const [queue, { refetch }] = createResource(async () => fetchQueue(20));

  async function skip(requestId: string) {
    await markQueueDone(requestId, "skipped");
    refetch();
  }

  return (
    <div>
      <h2
        style={{
          "font-size": "1.15rem",
          "font-weight": "600",
          "margin-bottom": "16px",
        }}
      >
        Labeling Queue
      </h2>

      <Show
        when={queue() && queue()!.length > 0}
        fallback={
          <div
            style={{
              padding: "24px",
              "text-align": "center",
              color: "var(--text-muted)",
              background: "var(--surface)",
              "border-radius": "var(--radius)",
              border: "1px solid var(--border)",
            }}
          >
            No pending labeling requests.
          </div>
        }
      >
        <div style={{ display: "flex", "flex-direction": "column", gap: "8px" }}>
          <For each={queue()!}>
            {(item) => (
              <div
                style={{
                  background: "var(--surface)",
                  border: "1px solid var(--border)",
                  "border-radius": "var(--radius)",
                  padding: "12px 16px",
                  display: "flex",
                  "align-items": "center",
                  "justify-content": "space-between",
                  gap: "12px",
                }}
              >
                <div>
                  <div style={{ "font-size": "0.9rem", "font-weight": "500" }}>
                    {new Date(item.bucket_start_ts).toLocaleString(undefined, {
                      month: "short",
                      day: "numeric",
                      hour: "2-digit",
                      minute: "2-digit",
                    })}
                  </div>
                  <div
                    style={{
                      "font-size": "0.8rem",
                      color: "var(--text-muted)",
                      "margin-top": "2px",
                    }}
                  >
                    {item.reason}
                    {item.predicted_label && ` · ${item.predicted_label}`}
                    {item.confidence !== null &&
                      ` · ${Math.round(item.confidence * 100)}%`}
                  </div>
                </div>
                <button
                  onClick={() => skip(item.request_id)}
                  style={{
                    padding: "6px 14px",
                    background: "var(--border)",
                    color: "var(--text-muted)",
                    border: "none",
                    "border-radius": "var(--radius)",
                    cursor: "pointer",
                    "font-size": "0.8rem",
                    "flex-shrink": "0",
                  }}
                >
                  Skip
                </button>
              </div>
            )}
          </For>
        </div>
      </Show>
    </div>
  );
};
