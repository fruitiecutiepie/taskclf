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
import { StatusProgress } from "./ui/StatusProgress";
import { StatusRow } from "./ui/StatusRow";
import { StatusSection } from "./ui/StatusSection";

export const TrainingPanel: Component<{
  trainState: Accessor<TrainState>;
}> = (props) => {
  const initToday = new Date().toISOString().slice(0, 10);
  const initFrom = (() => {
    const d = new Date();
    d.setDate(d.getDate() - 30);
    return d.toISOString().slice(0, 10);
  })();

  const [dateFrom, setDateFrom] = createSignal(initFrom);
  const [dateTo, setDateTo] = createSignal(initToday);
  const [boostRounds, setBoostRounds] = createSignal(100);
  const [classWeight, setClassWeight] = createSignal<"balanced" | "none">("balanced");
  const [synthetic, setSynthetic] = createSignal(false);

  const [dataCheck, setDataCheck] = createSignal<DataCheck | null>(null);
  const [checkedRange, setCheckedRange] = createSignal<{
    from: string;
    to: string;
  } | null>(null);
  const [models, setModels] = createSignal<ModelBundle[]>([]);
  const [checking, setChecking] = createSignal(false);
  const [checkError, setCheckError] = createSignal<string | null>(null);
  const [trainError, setTrainError] = createSignal<string | null>(null);
  const [submitting, setSubmitting] = createSignal(false);
  const [confirmPending, setConfirmPending] = createSignal(false);

  const ts = () => props.trainState();
  const isRunning = () => ts().status === "running";

  createEffect(
    on(
      () => [dateFrom(), dateTo()],
      () => {
        setDataCheck(null);
        setCheckedRange(null);
        setCheckError(null);
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

  const canTrain = createMemo(() => {
    if (synthetic()) {
      return true;
    }
    const dc = dataCheck();
    if (!dc) {
      return false;
    }
    return dc.trainable_rows > 0;
  });

  const trainDisabledReason = createMemo(() => {
    if (synthetic()) {
      return null;
    }
    const dc = dataCheck();
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
      setModels(models_sorted(ml));
    } catch {
      /* non-critical */
    }
  }

  async function data_check_submit() {
    if (checking()) {
      return;
    }
    setChecking(true);
    setCheckError(null);
    try {
      const [dc, ml] = await Promise.all([
        training_data_check(dateFrom(), dateTo()),
        models_list(),
      ]);
      setDataCheck(dc);
      setCheckedRange({ from: dateFrom(), to: dateTo() });
      setModels(models_sorted(ml));
    } catch (e: unknown) {
      setCheckError(e instanceof Error ? e.message : "Failed to check data");
    } finally {
      setChecking(false);
    }
  }

  models_refresh();

  async function training_submit() {
    if (submitting()) {
      return;
    }
    if (!confirmPending()) {
      setConfirmPending(true);
      return;
    }
    setConfirmPending(false);
    setSubmitting(true);
    setTrainError(null);
    try {
      await training_start({
        date_from: dateFrom(),
        date_to: dateTo(),
        num_boost_round: boostRounds(),
        class_weight: classWeight(),
        synthetic: synthetic(),
      });
    } catch (e: unknown) {
      setTrainError(e instanceof Error ? e.message : "Failed to start training");
    } finally {
      setSubmitting(false);
    }
  }

  async function training_cancel_submit() {
    try {
      await training_cancel();
    } catch (e: unknown) {
      setTrainError(e instanceof Error ? e.message : "Failed to cancel");
    }
  }

  const inputStyle = {
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

  const btnStyle = (
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
      <StatusSection title="Data Readiness" defaultOpen>
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
              value={dateFrom()}
              onInput={(e) => setDateFrom(e.currentTarget.value)}
              style={inputStyle}
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
              value={dateTo()}
              onInput={(e) => setDateTo(e.currentTarget.value)}
              style={inputStyle}
            />
          </div>
        </div>
        <button
          type="button"
          onClick={data_check_submit}
          disabled={checking()}
          style={btnStyle("ghost", checking())}
        >
          {checking() ? "Preparing…" : "Prepare Data"}
        </button>
        <Show when={dataCheck()}>
          <div style={{ "margin-top": "4px" }}>
            <Show when={(dataCheck()?.dates_built.length ?? 0) > 0}>
              <StatusRow
                label="built"
                value={`${dataCheck()?.dates_built.length} day(s) from AW`}
                color="#22c55e"
                tooltip="Days where feature data was freshly fetched from ActivityWatch"
              />
            </Show>
            <StatusRow
              label="features_days"
              value={`${dataCheck()?.dates_with_features.length ?? 0} / ${(dataCheck()?.dates_with_features.length ?? 0) + (dataCheck()?.dates_missing_features.length ?? 0)}`}
              tooltip="Days with activity data out of total days in range"
            />
            <StatusRow
              label="label_spans"
              value={String(dataCheck()?.label_span_count)}
              tooltip="Number of labeled time blocks in the selected range"
            />
            <StatusRow
              label="trainable_rows"
              value={
                (dataCheck()?.trainable_rows ?? 0) > 0
                  ? `${dataCheck()?.trainable_rows} (${dataCheck()?.trainable_labels.join(", ")})`
                  : "0"
              }
              color={(dataCheck()?.trainable_rows ?? 0) > 0 ? "#22c55e" : "#ef4444"}
              tooltip="Feature rows with a matching label — only these enter training"
            />
          </div>
        </Show>
        <Show when={checkError()}>
          <div
            style={{
              color: "#ef4444",
              "margin-top": "3px",
              "font-size": "0.58rem",
              display: "flex",
              "justify-content": "space-between",
              "align-items": "center",
            }}
          >
            <span>{checkError()}</span>
            <button
              type="button"
              onClick={() => setCheckError(null)}
              style={{
                cursor: "pointer",
                "margin-left": "4px",
                opacity: 0.7,
                background: "none",
                border: "none",
                color: "inherit",
                font: "inherit",
                padding: "0",
              }}
            >
              ✕
            </button>
          </div>
        </Show>
      </StatusSection>

      <StatusSection title="Train Model" defaultOpen>
        <div
          style={{
            display: "flex",
            "flex-direction": "column",
            gap: "4px",
            opacity: canTrain() || isRunning() ? 1 : 0.5,
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
                value={boostRounds()}
                onInput={(e) =>
                  setBoostRounds(parseInt(e.currentTarget.value, 10) || 100)
                }
                disabled={!canTrain() && !isRunning()}
                style={inputStyle}
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
                value={classWeight()}
                onChange={(e) =>
                  setClassWeight(e.currentTarget.value as "balanced" | "none")
                }
                disabled={!canTrain() && !isRunning()}
                style={inputStyle}
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
              onChange={(e) => setSynthetic(e.currentTarget.checked)}
              style={{ "accent-color": "#6366f1" }}
            />
            Synthetic data (no real data needed)
          </label>
          <Show when={checkedRange() && !synthetic()}>
            <div
              style={{ "font-size": "0.56rem", color: "#808080", "margin-top": "1px" }}
            >
              Training range: {checkedRange()?.from} — {checkedRange()?.to}
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
              when={!isRunning()}
              fallback={
                <button
                  type="button"
                  onClick={training_cancel_submit}
                  style={btnStyle("danger")}
                >
                  Cancel Training
                </button>
              }
            >
              <Show
                when={confirmPending()}
                fallback={
                  <button
                    type="button"
                    onClick={training_submit}
                    disabled={submitting() || !canTrain()}
                    style={btnStyle("primary", submitting() || !canTrain())}
                  >
                    {submitting() ? "Starting…" : "Train Model"}
                  </button>
                }
              >
                <button
                  type="button"
                  onClick={training_submit}
                  style={btnStyle("primary")}
                >
                  Confirm
                </button>
                <button
                  type="button"
                  onClick={() => setConfirmPending(false)}
                  style={btnStyle("ghost")}
                >
                  Cancel
                </button>
              </Show>
            </Show>
          </div>
          <Show when={trainDisabledReason() && !synthetic()}>
            <div
              style={{ "font-size": "0.56rem", color: "#808080", "margin-top": "1px" }}
            >
              {trainDisabledReason()}
            </div>
          </Show>
          <Show when={trainError()}>
            <div
              style={{
                color: "#ef4444",
                "margin-top": "3px",
                "font-size": "0.58rem",
                display: "flex",
                "justify-content": "space-between",
                "align-items": "center",
              }}
            >
              <span>{trainError()}</span>
              <button
                type="button"
                onClick={() => setTrainError(null)}
                style={{
                  cursor: "pointer",
                  "margin-left": "4px",
                  opacity: 0.7,
                  background: "none",
                  border: "none",
                  color: "inherit",
                  font: "inherit",
                  padding: "0",
                }}
              >
                ✕
              </button>
            </div>
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
          summaryColor={
            ts().status === "complete"
              ? "#22c55e"
              : ts().status === "failed"
                ? "#ef4444"
                : "#eab308"
          }
          defaultOpen
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
          <Show when={ts().error}>
            <StatusRow
              label="error"
              value={ts().error ?? ""}
              color="#ef4444"
              tooltip="Error message from the failed training run"
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
          style={{ ...btnStyle("ghost"), "margin-top": "4px" }}
        >
          Refresh
        </button>
      </StatusSection>
    </div>
  );
};
