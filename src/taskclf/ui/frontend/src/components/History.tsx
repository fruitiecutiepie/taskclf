import { type Component, createResource, For, Show } from "solid-js";
import { fetchLabels } from "../lib/api";

export const History: Component = () => {
  const [labels] = createResource(async () => fetchLabels(50));

  return (
    <div>
      <h2
        style={{
          "font-size": "1.15rem",
          "font-weight": "600",
          "margin-bottom": "16px",
        }}
      >
        Recent Labels
      </h2>

      <Show
        when={labels() && labels()!.length > 0}
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
            No labels yet.
          </div>
        }
      >
        <div
          style={{
            background: "var(--surface)",
            "border-radius": "var(--radius)",
            border: "1px solid var(--border)",
            overflow: "hidden",
          }}
        >
          <table
            style={{
              width: "100%",
              "border-collapse": "collapse",
              "font-size": "0.85rem",
            }}
          >
            <thead>
              <tr
                style={{
                  "border-bottom": "1px solid var(--border)",
                }}
              >
                {["Start", "End", "Label", "User", "Confidence", "Source"].map(
                  (h) => (
                    <th
                      style={{
                        "text-align": "left",
                        padding: "10px 12px",
                        color: "var(--text-muted)",
                        "font-weight": "500",
                        "font-size": "0.8rem",
                      }}
                    >
                      {h}
                    </th>
                  )
                )}
              </tr>
            </thead>
            <tbody>
              <For each={labels()!}>
                {(row) => (
                  <tr
                    style={{
                      "border-bottom": "1px solid var(--border)",
                    }}
                  >
                    <td style={{ padding: "8px 12px" }}>
                      {new Date(row.start_ts).toLocaleString(undefined, {
                        month: "short",
                        day: "numeric",
                        hour: "2-digit",
                        minute: "2-digit",
                      })}
                    </td>
                    <td style={{ padding: "8px 12px" }}>
                      {new Date(row.end_ts).toLocaleString(undefined, {
                        hour: "2-digit",
                        minute: "2-digit",
                      })}
                    </td>
                    <td
                      style={{
                        padding: "8px 12px",
                        "font-weight": "500",
                      }}
                    >
                      {row.label}
                    </td>
                    <td
                      style={{
                        padding: "8px 12px",
                        color: "var(--text-muted)",
                      }}
                    >
                      {row.user_id ?? "—"}
                    </td>
                    <td
                      style={{
                        padding: "8px 12px",
                        color: "var(--text-muted)",
                      }}
                    >
                      {row.confidence !== null
                        ? `${Math.round(row.confidence * 100)}%`
                        : "—"}
                    </td>
                    <td
                      style={{
                        padding: "8px 12px",
                        color: "var(--text-muted)",
                      }}
                    >
                      {row.provenance}
                    </td>
                  </tr>
                )}
              </For>
            </tbody>
          </table>
        </div>
      </Show>
    </div>
  );
};
