import {
  type Accessor,
  type Component,
  createEffect,
  createMemo,
  createSignal,
  For,
  on,
  Show,
} from "solid-js";
import {
  type DataCheck,
  type ModelBundle,
  models_list,
  training_cancel,
  training_data_check,
  training_start,
} from "../lib/api";
import type { TrainState } from "../lib/ws";
import { ErrorBanner } from "./ErrorBanner";
import { StatusProgress } from "./ui/StatusProgress";
import { StatusRow } from "./ui/StatusRow";
import { StatusSection } from "./ui/StatusSection";

export const TrainingPanel: Component<{
  train_state: Accessor<TrainState>;
}> = (props) => {
  const init_today = new Date().toISOString().slice(0, 10);
  const init_from = (() => {
    const d = new Date();
    d.setDate(d.getDate() - 30);
    return d.toISOString().slice(0, 10);
  })();

  const [date_from, set_date_from] = createSignal(init_from);
  const [date_to, set_date_to] = createSignal(init_today);
  const [boost_rounds, set_boost_rounds] = createSignal(100);
  const [class_weight, set_class_weight] = createSignal<"balanced" | "none">(
    "balanced",
  );
  const [synthetic, set_synthetic] = createSignal(false);

  const [data_check, set_data_check] = createSignal<DataCheck | null>(null);
  const [checked_range, set_checked_range] = createSignal<{
    from: string;
    to: string;
  } | null>(null);
  const [models, set_models] = createSignal<ModelBundle[]>([]);
  const [checking, set_checking] = createSignal(false);
  const [check_error, set_check_error] = createSignal<string | null>(null);
  const [train_error, set_train_error] = createSignal<string | null>(null);
  const [submitting, set_submitting] = createSignal(false);
  const [confirm_pending, set_confirm_pending] = createSignal(false);
  const [dismissed_run_error_key, set_dismissed_run_error_key] = createSignal<
    string | null
  >(null);

  const ts = () => props.train_state();
  const is_running = () => ts().status === "running";
  const run_error_key = () =>
    ts().error ? `${ts().job_id ?? "no-job"}:${ts().error}` : null;
  const visible_run_error = createMemo(() => {
    const error = ts().error;
    const key = run_error_key();
    if (!error || key === dismissed_run_error_key()) {
      return null;
    }
    return error;
  });

  createEffect(
    on(
      () => [date_from(), date_to()],
      () => {
        set_data_check(null);
        set_checked_range(null);
        set_check_error(null);
      },
      { defer: true },
    ),
  );

  createEffect(
    on(
      () => ts().status,
      (status, prevStatus) => {
        if (prevStatus === "running" && status === "complete") {
          models_refresh();
        }
      },
    ),
  );

  createEffect(
    on(
      run_error_key,
      (key) => {
        if (key === null) {
          set_dismissed_run_error_key(null);
        }
      },
      { defer: true },
    ),
  );

  const can_train = createMemo(() => {
    if (synthetic()) {
      return true;
    }
    const dc = data_check();
    if (!dc) {
      return false;
    }
    return dc.trainable_rows > 0;
  });

  const train_disabled_reason = createMemo(() => {
    if (synthetic()) {
      return null;
    }
    const dc = data_check();
    if (!dc) {
      return "Prepare data before training";
    }
    if (dc.dates_with_features.length === 0) {
      return "No feature data — is ActivityWatch running?";
    }
    if (dc.label_span_count === 0) {
      return "No label spans in selected range";
    }
    if (dc.trainable_rows === 0) {
      return "Labels don't overlap any feature windows — adjust labels or date range";
    }
    return null;
  });

  function models_sorted(ml: ModelBundle[]) {
    return ml
      .filter((m) => m.valid)
      .sort((a, b) => {
        if (a.created_at && b.created_at) {
          return b.created_at.localeCompare(a.created_at);
        }
        return 0;
      });
  }

  async function models_refresh() {
    try {
      const ml = await models_list();
      set_models(models_sorted(ml));
    } catch {
      /* non-critical */
    }
  }

  async function data_check_submit() {
    if (checking()) {
      return;
    }
    set_checking(true);
    set_check_error(null);
    try {
      const [dc, ml] = await Promise.all([
        training_data_check(date_from(), date_to()),
        models_list(),
      ]);
      set_data_check(dc);
      set_checked_range({ from: date_from(), to: date_to() });
      set_models(models_sorted(ml));
    } catch (e: unknown) {
      set_check_error(e instanceof Error ? e.message : "Failed to check data");
    } finally {
      set_checking(false);
    }
  }

  models_refresh();

  async function training_submit() {
    if (submitting()) {
      return;
    }
    if (!confirm_pending()) {
      set_confirm_pending(true);
      return;
    }
    set_confirm_pending(false);
    set_submitting(true);
    set_train_error(null);
    try {
      await training_start({
        date_from: date_from(),
        date_to: date_to(),
        num_boost_round: boost_rounds(),
        class_weight: class_weight(),
        synthetic: synthetic(),
      });
    } catch (e: unknown) {
      set_train_error(e instanceof Error ? e.message : "Failed to start training");
    } finally {
      set_submitting(false);
    }
  }

  async function training_cancel_submit() {
    try {
      await training_cancel();
    } catch (e: unknown) {
      set_train_error(e instanceof Error ? e.message : "Failed to cancel");
    }
  }

  const input_style = {
    width: "100%",
    padding: "3px 5px",
    background: "#252830",
    border: "1px solid #3a3d4a",
    "border-radius": "4px",
    color: "#e0e0e0",
    "font-size": "0.63rem",
    "font-family": "inherit",
    "box-sizing": "border-box" as const,
  };

  const btn_style = (
    variant: "primary" | "danger" | "ghost" = "primary",
    disabled = false,
  ) => ({
    padding: "4px 10px",
    border: "none",
    "border-radius": "5px",
    "font-size": "0.63rem",
    "font-family": "inherit",
    "font-weight": "600",
    cursor: disabled ? "not-allowed" : "pointer",
    transition: "all 0.15s ease",
    opacity: disabled ? 0.6 : 1,
    ...(variant === "primary"
      ? { background: "#6366f1", color: "#fff" }
      : variant === "danger"
        ? { background: "#ef4444", color: "#fff" }
        : { background: "#333", color: "#e0e0e0" }),
  });

  return (
    <div style={{ "font-size": "0.63rem" }}>
      <StatusSection title="Data Readiness" default_open>
        <div style={{ display: "flex", gap: "4px", "margin-bottom": "4px" }}>
          <div style={{ flex: 1 }}>
            <label
              for="train-date-from"
              style={{
                color: "#9a9a9a",
                "font-size": "0.58rem",
                display: "block",
                "margin-bottom": "1px",
              }}
            >
              From
            </label>
            <input
              id="train-date-from"
              type="date"
              value={date_from()}
              onInput={(e) => set_date_from(e.currentTarget.value)}
              style={input_style}
            />
          </div>
          <div style={{ flex: 1 }}>
            <label
              for="train-date-to"
              style={{
                color: "#9a9a9a",
                "font-size": "0.58rem",
                display: "block",
                "margin-bottom": "1px",
              }}
            >
              To
            </label>
            <input
              id="train-date-to"
              type="date"
              value={date_to()}
              onInput={(e) => set_date_to(e.currentTarget.value)}
              style={input_style}
            />
          </div>
        </div>
        <button
          type="button"
          onClick={data_check_submit}
          disabled={checking()}
          style={btn_style("ghost", checking())}
        >
          {checking() ? "Preparing…" : "Prepare Data"}
        </button>
        <Show when={data_check()}>
          <div style={{ "margin-top": "4px" }}>
            <Show when={(data_check()?.dates_built.length ?? 0) > 0}>
              <StatusRow
                label="built"
                value={`${data_check()?.dates_built.length} day(s) from AW`}
                color="#22c55e"
                tooltip="Days where feature data was freshly fetched from ActivityWatch"
              />
            </Show>
            <StatusRow
              label="features_days"
              value={`${data_check()?.dates_with_features.length ?? 0} / ${(data_check()?.dates_with_features.length ?? 0) + (data_check()?.dates_missing_features.length ?? 0)}`}
              tooltip="Days with activity data out of total days in range"
            />
            <StatusRow
              label="label_spans"
              value={String(data_check()?.label_span_count)}
              tooltip="Number of labeled time blocks in the selected range"
            />
            <StatusRow
              label="trainable_rows"
              value={
                (data_check()?.trainable_rows ?? 0) > 0
                  ? `${data_check()?.trainable_rows} (${data_check()?.trainable_labels.join(", ")})`
                  : "0"
              }
              color={(data_check()?.trainable_rows ?? 0) > 0 ? "#22c55e" : "#ef4444"}
              tooltip="Feature rows with a matching label — only these enter training"
            />
          </div>
        </Show>
        <Show when={check_error()}>
          <ErrorBanner
            message={check_error() ?? ""}
            on_close={() => set_check_error(null)}
          />
        </Show>
      </StatusSection>

      <StatusSection title="Train Model" default_open>
        <div
          style={{
            display: "flex",
            "flex-direction": "column",
            gap: "4px",
            opacity: can_train() || is_running() ? 1 : 0.5,
            transition: "opacity 0.2s ease",
          }}
        >
          <div style={{ display: "flex", gap: "4px", "align-items": "center" }}>
            <div style={{ flex: 1 }}>
              <label
                for="train-boost-rounds"
                style={{ color: "#9a9a9a", "font-size": "0.58rem" }}
              >
                Boost Rounds
              </label>
              <input
                id="train-boost-rounds"
                type="number"
                min="10"
                max="1000"
                step="10"
                value={boost_rounds()}
                onInput={(e) =>
                  set_boost_rounds(parseInt(e.currentTarget.value, 10) || 100)
                }
                disabled={!can_train() && !is_running()}
                style={input_style}
              />
            </div>
            <div style={{ flex: 1 }}>
              <label
                for="train-class-weight"
                style={{ color: "#9a9a9a", "font-size": "0.58rem" }}
              >
                Class Weight
              </label>
              <select
                id="train-class-weight"
                value={class_weight()}
                onChange={(e) =>
                  set_class_weight(e.currentTarget.value as "balanced" | "none")
                }
                disabled={!can_train() && !is_running()}
                style={input_style}
              >
                <option value="balanced">balanced</option>
                <option value="none">none</option>
              </select>
            </div>
          </div>
          <label
            style={{
              display: "flex",
              "align-items": "center",
              gap: "4px",
              color: "#9a9a9a",
              "font-size": "0.58rem",
              cursor: "pointer",
            }}
          >
            <input
              type="checkbox"
              checked={synthetic()}
              onChange={(e) => set_synthetic(e.currentTarget.checked)}
              style={{ "accent-color": "#6366f1" }}
            />
            Synthetic data (no real data needed)
          </label>
          <Show when={checked_range() && !synthetic()}>
            <div
              style={{ "font-size": "0.56rem", color: "#808080", "margin-top": "1px" }}
            >
              Training range: {checked_range()?.from} — {checked_range()?.to}
            </div>
          </Show>
          <div
            style={{
              display: "flex",
              gap: "4px",
              "align-items": "center",
              "margin-top": "2px",
            }}
          >
            <Show
              when={!is_running()}
              fallback={
                <button
                  type="button"
                  onClick={training_cancel_submit}
                  style={btn_style("danger")}
                >
                  Cancel Training
                </button>
              }
            >
              <Show
                when={confirm_pending()}
                fallback={
                  <button
                    type="button"
                    onClick={training_submit}
                    disabled={submitting() || !can_train()}
                    style={btn_style("primary", submitting() || !can_train())}
                  >
                    {submitting() ? "Starting…" : "Train Model"}
                  </button>
                }
              >
                <button
                  type="button"
                  onClick={training_submit}
                  style={btn_style("primary")}
                >
                  Confirm
                </button>
                <button
                  type="button"
                  onClick={() => set_confirm_pending(false)}
                  style={btn_style("ghost")}
                >
                  Cancel
                </button>
              </Show>
            </Show>
          </div>
          <Show when={train_disabled_reason() && !synthetic()}>
            <div
              style={{ "font-size": "0.56rem", color: "#808080", "margin-top": "1px" }}
            >
              {train_disabled_reason()}
            </div>
          </Show>
          <Show when={train_error()}>
            <ErrorBanner
              message={train_error() ?? ""}
              on_close={() => set_train_error(null)}
            />
          </Show>
        </div>
      </StatusSection>

      <Show when={ts().status !== "idle"}>
        <StatusSection
          title="Training Progress"
          summary={
            ts().status === "running"
              ? `${ts().progress_pct ?? 0}%`
              : ts().status === "complete"
                ? "done"
                : ts().status === "failed"
                  ? "failed"
                  : ""
          }
          summary_color={
            ts().status === "complete"
              ? "#22c55e"
              : ts().status === "failed"
                ? "#ef4444"
                : "#eab308"
          }
          default_open
        >
          <StatusRow
            label="status"
            value={ts().status}
            color={
              ts().status === "complete"
                ? "#22c55e"
                : ts().status === "failed"
                  ? "#ef4444"
                  : "#eab308"
            }
            tooltip="Current state of the training job"
          />
          <Show when={ts().step}>
            <StatusRow
              label="step"
              value={ts().step ?? ""}
              dim
              tooltip="Current step in the training pipeline"
            />
          </Show>
          <Show when={ts().message}>
            <StatusRow
              label="message"
              value={ts().message ?? ""}
              dim
              tooltip="Latest progress message from the trainer"
            />
          </Show>
          <Show when={ts().progress_pct != null && ts().status === "running"}>
            <StatusProgress pct={ts().progress_pct ?? 0} />
          </Show>
          <Show when={visible_run_error()}>
            <ErrorBanner
              message={visible_run_error() ?? ""}
              on_close={() => set_dismissed_run_error_key(run_error_key())}
            />
          </Show>
          <Show when={ts().metrics}>
            <StatusRow
              label="macro_f1"
              value={ts().metrics?.macro_f1?.toFixed(3) ?? "—"}
              color="#22c55e"
              tooltip="F1 score averaged equally across all classes — good for imbalanced datasets"
            />
            <StatusRow
              label="weighted_f1"
              value={ts().metrics?.weighted_f1?.toFixed(3) ?? "—"}
              color="#22c55e"
              tooltip="F1 score weighted by class frequency — reflects overall accuracy"
            />
          </Show>
          <Show when={ts().model_dir}>
            <StatusRow
              label="model_dir"
              value={ts().model_dir?.split("/").pop() ?? ts().model_dir ?? ""}
              dim
              mono
              tooltip="Output directory for the trained model bundle"
            />
          </Show>
        </StatusSection>
      </Show>

      <StatusSection title="Models" summary={`${models().length}`}>
        <Show
          when={models().length > 0}
          fallback={
            <StatusRow
              label="status"
              value="no models found"
              dim
              tooltip="No trained model bundles found — train a model first"
            />
          }
        >
          <For each={models()}>
            {(m) => (
              <div
                style={{
                  padding: "3px 0",
                  "border-bottom": "1px solid #2a2a2a",
                }}
              >
                <StatusRow
                  label="id"
                  value={m.model_id}
                  mono
                  tooltip="Unique identifier for this model bundle"
                />
                <Show when={m.macro_f1 != null}>
                  <StatusRow
                    label="macro_f1"
                    value={m.macro_f1?.toFixed(3) ?? ""}
                    color="#22c55e"
                    tooltip="Model's macro-averaged F1 score on the validation set"
                  />
                </Show>
                <Show when={m.created_at}>
                  <StatusRow
                    label="created"
                    value={m.created_at?.slice(0, 19).replace("T", " ") ?? ""}
                    dim
                    tooltip="When this model was trained"
                  />
                </Show>
              </div>
            )}
          </For>
        </Show>
        <button
          type="button"
          onClick={models_refresh}
          style={{ ...btn_style("ghost"), "margin-top": "4px" }}
        >
          Refresh
        </button>
      </StatusSection>
    </div>
  );
};
