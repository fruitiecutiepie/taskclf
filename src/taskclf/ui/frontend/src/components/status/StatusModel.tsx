import {
  type Accessor,
  type Component,
  createEffect,
  createMemo,
  createSignal,
  on,
  Show,
} from "solid-js";
import {
  type CurrentModelBundleInspectResponse,
  model_bundle_inspect_current,
} from "../../lib/api";
import { path_trunc } from "../../lib/format";
import { LABEL_COLORS } from "../../lib/labelColors";
import type { TrayState } from "../../lib/ws";
import { StatusRow } from "../ui/StatusRow";
import { StatusSection } from "../ui/StatusSection";

export const StatusModel: Component<{
  tray_state: Accessor<TrayState>;
}> = (props) => {
  const t = () => props.tray_state();

  const [bundle_inspect, set_bundle_inspect] =
    createSignal<CurrentModelBundleInspectResponse | null>(null);
  const [bundle_inspect_error, set_bundle_inspect_error] = createSignal<string | null>(
    null,
  );

  createEffect(
    on(
      () => [t().model_loaded, t().model_dir] as const,
      async ([loaded, dir]) => {
        if (!loaded || !dir) {
          set_bundle_inspect(null);
          set_bundle_inspect_error(null);
          return;
        }
        try {
          const r = await model_bundle_inspect_current();
          set_bundle_inspect(r);
          set_bundle_inspect_error(null);
        } catch (e: unknown) {
          set_bundle_inspect(null);
          set_bundle_inspect_error(
            e instanceof Error ? e.message : "Bundle inspect failed",
          );
        }
      },
      { defer: true },
    ),
  );

  const summary = createMemo(() => (t().model_loaded ? "loaded" : "not loaded"));
  const summary_color = createMemo(() => (t().model_loaded ? "#22c55e" : "#ef4444"));

  const loaded_bundle_inspect = createMemo(() => {
    const b = bundle_inspect();
    if (b?.loaded) {
      return b;
    }
    return undefined;
  });

  return (
    <StatusSection title="Model" summary={summary()} summary_color={summary_color()}>
      <StatusRow
        label="loaded"
        value={t().model_loaded ? "yes" : "no"}
        color={t().model_loaded ? "#22c55e" : "#ef4444"}
        tooltip="Whether a trained model is currently loaded for inference"
      />
      <Show when={t().model_dir}>
        {(dir) => (
          <StatusRow
            label="model_dir"
            value={path_trunc(dir())}
            dim
            mono
            tooltip="Directory path of the loaded model bundle"
          />
        )}
      </Show>
      <Show when={t().model_schema_hash}>
        {(hash) => (
          <StatusRow
            label="schema_hash"
            value={hash()}
            dim
            mono
            tooltip="Feature schema hash the model was trained with — must match current schema to run inference"
          />
        )}
      </Show>
      <Show when={t().suggested_label}>
        {(label) => (
          <>
            <StatusRow
              label="suggested"
              value={label()}
              color={LABEL_COLORS[label()] ?? "#e0e0e0"}
              tooltip="Label the model suggests for the current activity block"
            />
            <StatusRow
              label="suggestion_conf"
              value={`${Math.round((t().suggested_confidence ?? 0) * 100)}%`}
              tooltip="Confidence of the current label suggestion"
            />
          </>
        )}
      </Show>
      <Show when={!t().suggested_label}>
        <StatusRow
          label="suggested"
          value="none"
          dim
          tooltip="Label the model suggests for the current activity block"
        />
      </Show>
      <Show when={bundle_inspect_error()}>
        <StatusRow
          label="inspect"
          value={bundle_inspect_error() ?? ""}
          color="#ef4444"
          dim
          tooltip="Bundle-saved validation metrics (tray API)"
        />
      </Show>
      <Show
        when={
          bundle_inspect()
          && bundle_inspect()?.loaded === false
          && !bundle_inspect_error()
        }
      >
        <StatusRow
          label="inspect"
          value={
            bundle_inspect()?.loaded === false
              ? (bundle_inspect() as { reason: string }).reason.replaceAll("_", " ")
              : ""
          }
          dim
          tooltip="Why bundle inspection is unavailable — e.g. no model path from tray"
        />
      </Show>
      <Show when={loaded_bundle_inspect}>
        {(i) => {
          const row = i as Extract<CurrentModelBundleInspectResponse, { loaded: true }>;
          const meta = row.metadata as Record<string, string>;
          const top_pairs = row.bundle_saved_validation.top_confusion_pairs.slice(0, 3);
          const top_str =
            top_pairs.length === 0
              ? "—"
              : top_pairs
                  .map((p) => `${p.true_label}→${p.pred_label} (${p.count})`)
                  .join(", ");
          return (
            <>
              <StatusRow
                label="val_macro_f1"
                value={row.bundle_saved_validation.macro_f1.toFixed(4)}
                color="#22c55e"
                tooltip="Macro F1 on the validation split saved in the bundle (not held-out test)"
              />
              <StatusRow
                label="val_weighted_f1"
                value={row.bundle_saved_validation.weighted_f1.toFixed(4)}
                color="#22c55e"
                tooltip="Weighted F1 on the validation split saved in the bundle"
              />
              <Show when={meta.train_date_from && meta.train_date_to}>
                <StatusRow
                  label="trained"
                  value={`${meta.train_date_from} — ${meta.train_date_to}`}
                  dim
                  tooltip="Training date range from bundle metadata"
                />
              </Show>
              <StatusRow
                label="top_confusions"
                value={top_str}
                dim
                tooltip="Largest off-diagonal confusion counts from the saved validation matrix"
              />
            </>
          );
        }}
      </Show>
    </StatusSection>
  );
};
