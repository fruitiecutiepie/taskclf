import {
  type Accessor,
  type Component,
  For,
  Show,
  createEffect,
  createMemo,
  createSignal,
  on,
} from "solid-js";
import type { TrainState } from "../lib/ws";
import {
  type DataCheck,
  type ModelBundle,
  cancelTraining,
  listModels,
  startTraining,
  trainDataCheck,
} from "../lib/api";
import { StatusRow } from "./ui/StatusRow";
import { StatusSection } from "./ui/StatusSection";
import { StatusProgress } from "./ui/StatusProgress";

export const TrainingPanel: Component<{
  trainState: Accessor<TrainState>;
}> = (props) => {
  const initToday = new Date().toISOString().slice(0, 10);
  const initFrom = (() => { const d = new Date(); d.setDate(d.getDate() - 30); return d.toISOString().slice(0, 10); })();

  const [dateFrom, setDateFrom] = createSignal(initFrom);
  const [dateTo, setDateTo] = createSignal(initToday);
  const [boostRounds, setBoostRounds] = createSignal(100);
  const [classWeight, setClassWeight] = createSignal<"balanced" | "none">("balanced");
  const [synthetic, setSynthetic] = createSignal(false);

  const [dataCheck, setDataCheck] = createSignal<DataCheck | null>(null);
  const [checkedRange, setCheckedRange] = createSignal<{ from: string; to: string } | null>(null);
  const [models, setModels] = createSignal<ModelBundle[]>([]);
  const [checking, setChecking] = createSignal(false);
  const [checkError, setCheckError] = createSignal<string | null>(null);
  const [trainError, setTrainError] = createSignal<string | null>(null);
  const [submitting, setSubmitting] = createSignal(false);
  const [confirmPending, setConfirmPending] = createSignal(false);

  const ts = () => props.trainState();
  const isRunning = () => ts().status === "running";

  createEffect(on(
    () => [dateFrom(), dateTo()],
    () => { setDataCheck(null); setCheckedRange(null); setCheckError(null); },
    { defer: true },
  ));

  createEffect(on(
    () => ts().status,
    (status, prevStatus) => {
      if (prevStatus === "running" && status === "complete") {
        refreshModels();
      }
    },
  ));

  const canTrain = createMemo(() => {
    if (synthetic()) return true;
    const dc = dataCheck();
    if (!dc) return false;
    return dc.trainable_rows > 0;
  });

  const trainDisabledReason = createMemo(() => {
    if (synthetic()) return null;
    if (!dataCheck()) return "Prepare data before training";
    const dc = dataCheck()!;
    if (dc.dates_with_features.length === 0)
      return "No feature data — is ActivityWatch running?";
    if (dc.label_span_count === 0)
      return "No label spans in selected range";
    if (dc.trainable_rows === 0)
      return "Labels don't overlap any feature windows — adjust labels or date range";
    return null;
  });

  function sortModels(ml: ModelBundle[]) {
    return ml
      .filter((m) => m.valid)
      .sort((a, b) => {
        if (a.created_at && b.created_at) return b.created_at.localeCompare(a.created_at);
        return 0;
      });
  }

  async function refreshModels() {
    try {
      const ml = await listModels();
      setModels(sortModels(ml));
    } catch { /* non-critical */ }
  }

  async function handleCheckData() {
    if (checking()) return;
    setChecking(true);
    setCheckError(null);
    try {
      const [dc, ml] = await Promise.all([
        trainDataCheck(dateFrom(), dateTo()),
        listModels(),
      ]);
      setDataCheck(dc);
      setCheckedRange({ from: dateFrom(), to: dateTo() });
      setModels(sortModels(ml));
    } catch (e: any) {
      setCheckError(e?.message ?? "Failed to check data");
    } finally {
      setChecking(false);
    }
  }

  refreshModels();

  async function handleTrain() {
    if (submitting()) return;
    if (!confirmPending()) {
      setConfirmPending(true);
      return;
    }
    setConfirmPending(false);
    setSubmitting(true);
    setTrainError(null);
    try {
      await startTraining({
        date_from: dateFrom(),
        date_to: dateTo(),
        num_boost_round: boostRounds(),
        class_weight: classWeight(),
        synthetic: synthetic(),
      });
    } catch (e: any) {
      setTrainError(e?.message ?? "Failed to start training");
    } finally {
      setSubmitting(false);
    }
  }

  async function handleCancel() {
    try {
      await cancelTraining();
    } catch (e: any) {
      setTrainError(e?.message ?? "Failed to cancel");
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

  const btnStyle = (variant: "primary" | "danger" | "ghost" = "primary", disabled = false) => ({
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
            <label style={{ color: "#9a9a9a", "font-size": "0.58rem", display: "block", "margin-bottom": "1px" }}>From</label>
            <input
              type="date"
              value={dateFrom()}
              onInput={(e) => setDateFrom(e.currentTarget.value)}
              style={inputStyle}
            />
          </div>
          <div style={{ flex: 1 }}>
            <label style={{ color: "#9a9a9a", "font-size": "0.58rem", display: "block", "margin-bottom": "1px" }}>To</label>
            <input
              type="date"
              value={dateTo()}
              onInput={(e) => setDateTo(e.currentTarget.value)}
              style={inputStyle}
            />
          </div>
        </div>
        <button
          onClick={handleCheckData}
          disabled={checking()}
          style={btnStyle("ghost", checking())}
        >
          {checking() ? "Preparing…" : "Prepare Data"}
        </button>
        <Show when={dataCheck()}>
          <div style={{ "margin-top": "4px" }}>
            <Show when={dataCheck()!.dates_built.length > 0}>
              <StatusRow label="built" value={`${dataCheck()!.dates_built.length} day(s) from AW`} color="#22c55e" tooltip="Days where feature data was freshly fetched from ActivityWatch" />
            </Show>
            <StatusRow label="features_days" value={`${dataCheck()!.dates_with_features.length} / ${dataCheck()!.dates_with_features.length + dataCheck()!.dates_missing_features.length}`} tooltip="Days with activity data out of total days in range" />
            <StatusRow label="label_spans" value={String(dataCheck()!.label_span_count)} tooltip="Number of labeled time blocks in the selected range" />
            <StatusRow
              label="trainable_rows"
              value={dataCheck()!.trainable_rows > 0
                ? `${dataCheck()!.trainable_rows} (${dataCheck()!.trainable_labels.join(", ")})`
                : "0"}
              color={dataCheck()!.trainable_rows > 0 ? "#22c55e" : "#ef4444"}
              tooltip="Feature rows with a matching label — only these enter training"
            />
          </div>
        </Show>
        <Show when={checkError()}>
          <div
            style={{ color: "#ef4444", "margin-top": "3px", "font-size": "0.58rem", display: "flex", "justify-content": "space-between", "align-items": "center" }}
          >
            <span>{checkError()}</span>
            <span onClick={() => setCheckError(null)} style={{ cursor: "pointer", "margin-left": "4px", opacity: 0.7 }}>✕</span>
          </div>
        </Show>
      </StatusSection>

      <StatusSection title="Train Model" defaultOpen>
        <div style={{
          display: "flex",
          "flex-direction": "column",
          gap: "4px",
          opacity: canTrain() || isRunning() ? 1 : 0.5,
          transition: "opacity 0.2s ease",
        }}>
          <div style={{ display: "flex", gap: "4px", "align-items": "center" }}>
            <div style={{ flex: 1 }}>
              <label style={{ color: "#9a9a9a", "font-size": "0.58rem" }}>Boost Rounds</label>
              <input
                type="number"
                min="10"
                max="1000"
                step="10"
                value={boostRounds()}
                onInput={(e) => setBoostRounds(parseInt(e.currentTarget.value) || 100)}
                disabled={!canTrain() && !isRunning()}
                style={inputStyle}
              />
            </div>
            <div style={{ flex: 1 }}>
              <label style={{ color: "#9a9a9a", "font-size": "0.58rem" }}>Class Weight</label>
              <select
                value={classWeight()}
                onChange={(e) => setClassWeight(e.currentTarget.value as "balanced" | "none")}
                disabled={!canTrain() && !isRunning()}
                style={inputStyle}
              >
                <option value="balanced">balanced</option>
                <option value="none">none</option>
              </select>
            </div>
          </div>
          <label style={{ display: "flex", "align-items": "center", gap: "4px", color: "#9a9a9a", "font-size": "0.58rem", cursor: "pointer" }}>
            <input
              type="checkbox"
              checked={synthetic()}
              onChange={(e) => setSynthetic(e.currentTarget.checked)}
              style={{ "accent-color": "#6366f1" }}
            />
            Synthetic data (no real data needed)
          </label>
          <Show when={checkedRange() && !synthetic()}>
            <div style={{ "font-size": "0.56rem", color: "#808080", "margin-top": "1px" }}>
              Training range: {checkedRange()!.from} — {checkedRange()!.to}
            </div>
          </Show>
          <div style={{ display: "flex", gap: "4px", "align-items": "center", "margin-top": "2px" }}>
            <Show
              when={!isRunning()}
              fallback={
                <button onClick={handleCancel} style={btnStyle("danger")}>Cancel Training</button>
              }
            >
              <Show when={confirmPending()} fallback={
                <button
                  onClick={handleTrain}
                  disabled={submitting() || !canTrain()}
                  style={btnStyle("primary", submitting() || !canTrain())}
                >
                  {submitting() ? "Starting…" : "Train Model"}
                </button>
              }>
                <button
                  onClick={handleTrain}
                  style={btnStyle("primary")}
                >
                  Confirm
                </button>
                <button
                  onClick={() => setConfirmPending(false)}
                  style={btnStyle("ghost")}
                >
                  Cancel
                </button>
              </Show>
            </Show>
          </div>
          <Show when={trainDisabledReason() && !synthetic()}>
            <div style={{ "font-size": "0.56rem", color: "#808080", "margin-top": "1px" }}>
              {trainDisabledReason()}
            </div>
          </Show>
          <Show when={trainError()}>
            <div
              style={{ color: "#ef4444", "margin-top": "3px", "font-size": "0.58rem", display: "flex", "justify-content": "space-between", "align-items": "center" }}
            >
              <span>{trainError()}</span>
              <span onClick={() => setTrainError(null)} style={{ cursor: "pointer", "margin-left": "4px", opacity: 0.7 }}>✕</span>
            </div>
          </Show>
        </div>
      </StatusSection>

      <Show when={ts().status !== "idle"}>
        <StatusSection
          title="Training Progress"
          summary={
            ts().status === "running" ? `${ts().progress_pct ?? 0}%` :
            ts().status === "complete" ? "done" :
            ts().status === "failed" ? "failed" : ""
          }
          summaryColor={
            ts().status === "complete" ? "#22c55e" :
            ts().status === "failed" ? "#ef4444" :
            "#eab308"
          }
          defaultOpen
        >
          <StatusRow
            label="status"
            value={ts().status}
            color={
              ts().status === "complete" ? "#22c55e" :
              ts().status === "failed" ? "#ef4444" :
              "#eab308"
            }
            tooltip="Current state of the training job"
          />
          <Show when={ts().step}>
            <StatusRow label="step" value={ts().step!} dim tooltip="Current step in the training pipeline" />
          </Show>
          <Show when={ts().message}>
            <StatusRow label="message" value={ts().message!} dim tooltip="Latest progress message from the trainer" />
          </Show>
          <Show when={ts().progress_pct != null && ts().status === "running"}>
            <StatusProgress pct={ts().progress_pct!} />
          </Show>
          <Show when={ts().error}>
            <StatusRow label="error" value={ts().error!} color="#ef4444" tooltip="Error message from the failed training run" />
          </Show>
          <Show when={ts().metrics}>
            <StatusRow
              label="macro_f1"
              value={((ts().metrics as any)?.macro_f1 as number)?.toFixed(3) ?? "—"}
              color="#22c55e"
              tooltip="F1 score averaged equally across all classes — good for imbalanced datasets"
            />
            <StatusRow
              label="weighted_f1"
              value={((ts().metrics as any)?.weighted_f1 as number)?.toFixed(3) ?? "—"}
              color="#22c55e"
              tooltip="F1 score weighted by class frequency — reflects overall accuracy"
            />
          </Show>
          <Show when={ts().model_dir}>
            <StatusRow label="model_dir" value={ts().model_dir!.split("/").pop() ?? ts().model_dir!} dim mono tooltip="Output directory for the trained model bundle" />
          </Show>
        </StatusSection>
      </Show>

      <StatusSection title="Models" summary={`${models().length}`}>
        <Show
          when={models().length > 0}
          fallback={<StatusRow label="status" value="no models found" dim tooltip="No trained model bundles found — train a model first" />}
        >
          <For each={models()}>
            {(m) => (
              <div style={{
                padding: "3px 0",
                "border-bottom": "1px solid #2a2a2a",
              }}>
                <StatusRow label="id" value={m.model_id} mono tooltip="Unique identifier for this model bundle" />
                <Show when={m.macro_f1 != null}>
                  <StatusRow label="macro_f1" value={m.macro_f1!.toFixed(3)} color="#22c55e" tooltip="Model's macro-averaged F1 score on the validation set" />
                </Show>
                <Show when={m.created_at}>
                  <StatusRow label="created" value={m.created_at!.slice(0, 19).replace("T", " ")} dim tooltip="When this model was trained" />
                </Show>
              </div>
            )}
          </For>
        </Show>
        <button
          onClick={refreshModels}
          style={{ ...btnStyle("ghost"), "margin-top": "4px" }}
        >
          Refresh
        </button>
      </StatusSection>
    </div>
  );
};
