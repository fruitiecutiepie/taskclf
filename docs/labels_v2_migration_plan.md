# Labels v2 Migration Plan

This document outlines the migration plan to `labels_v2` and highlights relevant touchpoints in the codebase for each phase.
It is aligned to the current contract where raw telemetry flows through `EvidenceSnapshot`, deterministic observed facts (`ObservedLabel`), and then semantic interpretation (`SemanticLabel`) persisted in a `LabelEnvelope`.

---

## Phase 1: Core Types & Uncertainty Structures
*The foundational data model update.*

**Tasks:**
- [x] Freeze the existing `Mode` type (Produce, Consume, Coordinate, Attend, Idle).
- [x] Add the new `Subtype` values (`Analyze`, `Learn`, `ExploreReference`, `Monitor`) and keep persona/domain concepts such as `Design`, `Analysis`, and `Operations` in `OutputDomain` rather than `Subtype`.
- [x] Explicitly define `InteractionStyle`, `CollaborationMode`, and `OutputDomain` (distinct from app used).
- [x] Introduce `AxisDecision<T>` to wrap axes with `value`, `confidence`, `alternatives`, and `reason_codes`.
- [x] Define the explicit semantic provenance fields: `SupportState`, `IntentBasis`, `ModeSource`, and the override path.
- [x] Create the new `ObservedLabel`, `SemanticLabel`, and `LabelEnvelope` interfaces, with `LabelEnvelope` carrying `observed?` and `semantic`.

**Codebase Touchpoints:**
- `src/taskclf/core/types.py`: Currently defines `CoreLabel` as `StrEnum` and `LABEL_SET_V1`. This is where `Mode`, `Subtype`, `OutputDomain`, `InteractionStyle`, `CollaborationMode`, `AxisDecision`, `SupportState`, `IntentBasis`, `ModeSource`, `ObservedLabel`, `SemanticLabel`, and `LabelEnvelope` need to be defined.
- `schema/labels_v1.json`: The JSON schema for the old labels will need a `labels_v2.json` counterpart.

---

## Phase 2: Evidence, Observed, and Inference Layers
*Decoupling raw factual observation, deterministic observed facts, and resulting semantic labels.*

**Tasks:**
- [x] Define the `EvidenceSnapshot` object with raw observational signals (foreground ms, key events, active mic, etc.).
- [x] Define the deterministic projection from `EvidenceSnapshot` to `ObservedLabel`.
- [x] Extract the pipeline responsible for computing 15s–60s "Evidence windows".
- [x] Plumb the pipeline that feeds these Evidence windows into deterministic `ObservedLabel` records and then into 2m–5m "Inference windows" where semantic labeling occurs.

**Codebase Touchpoints:**
- `src/taskclf/core/types.py`: Defines `FeatureRowBase` and `FeatureRow` which represent bucketed observations (typically 60s). These rows will either map to or serve as the substrate for `EvidenceSnapshot`, which then projects deterministically into `ObservedLabel`.
- `src/taskclf/infer/batch.py` / `src/taskclf/infer/online.py`: Current inference pipelines operate directly on `FeatureRow`s using LightGBM. These pipelines need to be adapted to separate evidence collection, observed-fact projection, and semantic inference.

---

## Phase 3: Decision Engine & Precedence Rules
*The deterministic rules that consume `ObservedLabel` plus `EvidenceSnapshot` context and output `SemanticLabel`.*

**Tasks:**
- [x] Implement the 5-rule `Mode` decision order: Idle → Attend → Coordinate → Produce → Consume.
- [x] Implement the tie-break conflict logic (e.g., Produce vs Consume, Coordinate vs Attend) and assign semantic provenance such as `intent_basis`, `mode_source`, and `support_state` when the observed layer alone does not settle the choice.
- [x] Add the subtype policy to restrict available subtypes based on the dominant Mode.

**Codebase Touchpoints:**
- `src/taskclf/infer/baseline.py`: Contains heuristic baseline predictors (`predict_baseline`) which can be adapted to serve as the new deterministic precedence engine.
- `src/taskclf/infer/prediction.py`: Contains the `WindowPrediction` models, which will be updated to output the new semantic label structure (typically persisted inside `LabelEnvelope` with optional `observed` facts) rather than a flat string.

---

## Phase 4: Domain Plugins & Escalation
*Extensibility for different personas and edge cases.*

**Tasks:**
- [x] Define the base `PluginPayload` container for extensibility.
- [x] Draft initial empty/placeholder schemas for standard plugins (Software, Research, Design, Education, Operations, Analysis).
- [x] Implement the escalation trigger logic for throwing `Unknown` or `MixedUnknown` statuses.

**Codebase Touchpoints:**
- `src/taskclf/core/types.py`: Likely location to define the plugin types (e.g., `SoftwareLabels`, `ResearchLabels`) and `PluginPayload`.
- `src/taskclf/infer/batch.py` and `src/taskclf/infer/online.py`: Contains `MIXED_UNKNOWN` thresholding (`reject_threshold`). This logic will be enhanced with the new escalation rules for missing plugins or subtypes.

---

## Phase 5: Session Aggregation
*Rolling up smaller chunks into reportable sessions.*

**Tasks:**
- [x] Implement the logic grouping inference windows into 15m–90m "Session Aggregates".
- [x] Apply the `60% dominant mode` aggregation rule (otherwise resolving to `Mixed`).
- [x] Evaluate and map alignment of optional axes during aggregation.

**Codebase Touchpoints:**
- `src/taskclf/infer/batch.py`: Contains smoothing and merging logic (`rolling_majority`, `segmentize`, `merge_short_segments`) which handles grouping contiguous chunks. This will need to be refactored to implement the 60% dominance rule.
- `src/taskclf/core/types.py`: `LabelSpan` defines contiguous time spans for labels. This may need updating or extending to support `SessionMode` aggregates.

---

## Phase 6: Playbook & Migration
*Tying it together and back-porting existing data.*

**Tasks:**
- [x] Draft the "Annotation playbook" containing canonical/boundary examples and the dominance checklist for the team.
- [x] Write a data migration/mapping script to convert `v1` datasets to `labels_v2` `LabelEnvelope` records, ensuring backward compatibility.
- [x] Treat migrated records without telemetry backfill as semantic-only legacy records; native `labels_v2` records should carry both `observed` and `semantic`.

**Codebase Touchpoints:**
- `docs/guide/labels_v1.md`: Create a new `labels_v2_playbook.md` to serve as the annotation guide.
- New scripts/files needed in `src/taskclf/data/` or `src/taskclf/cli/` to migrate existing datasets containing `LABEL_SET_V1` strings into `LabelEnvelope` structured records, backfilling `observed` when telemetry exists and otherwise emitting semantic-first legacy records with conservative provenance (`intent_basis`, `mode_source`, `support_state`).
