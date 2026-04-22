# Labels v2 Migration Plan

This document outlines the migration plan to `labels_v2` and highlights relevant touchpoints in the codebase for each phase.

---

## Phase 1: Core Types & Uncertainty Structures
*The foundational data model update.*

**Tasks:**
- [x] Freeze the existing `Mode` type (Produce, Consume, Coordinate, Attend, Idle).
- [x] Add the new `Subtype` values (`Analyze`, `Learn`, `ExploreReference`, `Monitor`, `Design`, `Analysis`, `Operations`).
- [x] Explicitly define definitions for `InteractionStyle`, `CollaborationMode`, and `OutputDomain` (distinct from app used).
- [x] Introduce `AxisDecision<T>` to wrap axes with `value`, `confidence`, `alternatives`, and `reason_codes`.
- [x] Define the explicit `SupportState` semantic enum.
- [x] Create the new `LabelEnvelope` and `CrossDomainLabel` interfaces.

**Codebase Touchpoints:**
- `src/taskclf/core/types.py`: Currently defines `CoreLabel` as `StrEnum` and `LABEL_SET_V1`. This is where `Mode`, `Subtype`, `OutputDomain`, `InteractionStyle`, `CollaborationMode`, `AxisDecision`, and `SupportState` need to be defined.
- `schema/labels_v1.json`: The JSON schema for the old labels will need a `labels_v2.json` counterpart.

---

## Phase 2: Evidence vs. Inference Layer
*Decoupling the raw factual observation from the resulting label.*

**Tasks:**
- [x] Define the `EvidenceSnapshot` object with raw observational signals (foreground ms, key events, active mic, etc.).
- [x] Extract the pipeline responsible for computing 15s–60s "Evidence windows".
- [x] Plumb the pipeline that feeds these Evidence windows into 2m–5m "Inference windows" where semantic labeling occurs.

**Codebase Touchpoints:**
- `src/taskclf/core/types.py`: Defines `FeatureRowBase` and `FeatureRow` which represent bucketed observations (typically 60s). These rows will either map to or serve as the substrate for `EvidenceSnapshot`.
- `src/taskclf/infer/batch.py` / `src/taskclf/infer/online.py`: Current inference pipelines operate directly on `FeatureRow`s using LightGBM. These pipelines need to be adapted to separate the Evidence collection from the semantic Inference step.

---

## Phase 3: Decision Engine & Precedence Rules
*The deterministic rules that consume Evidence and output Inference.*

**Tasks:**
- [ ] Implement the 5-rule `Mode` decision order: Idle → Attend → Coordinate → Produce → Consume.
- [ ] Implement the tie-break conflict logic (e.g., Produce vs Consume, Coordinate vs Attend).
- [ ] Add the subtype policy to restrict available subtypes based on the dominant Mode.

**Codebase Touchpoints:**
- `src/taskclf/infer/baseline.py`: Contains heuristic baseline predictors (`predict_baseline`) which can be adapted to serve as the new deterministic precedence engine.
- `src/taskclf/infer/prediction.py`: Contains the `WindowPrediction` models, which will be updated to output the new `CrossDomainLabel` format rather than a flat string.

---

## Phase 4: Domain Plugins & Escalation
*Extensibility for different personas and edge cases.*

**Tasks:**
- [ ] Define the base `PluginPayload` container for extensibility.
- [ ] Draft initial empty/placeholder schemas for standard plugins (Software, Research, Design, Education, Operations, Analysis).
- [ ] Implement the escalation trigger logic for throwing `Unknown` or `MixedUnknown` statuses.

**Codebase Touchpoints:**
- `src/taskclf/core/types.py`: Likely location to define the plugin types (e.g., `SoftwareLabels`, `ResearchLabels`) and `PluginPayload`.
- `src/taskclf/infer/batch.py` and `src/taskclf/infer/online.py`: Contains `MIXED_UNKNOWN` thresholding (`reject_threshold`). This logic will be enhanced with the new escalation rules for missing plugins or subtypes.

---

## Phase 5: Session Aggregation
*Rolling up smaller chunks into reportable sessions.*

**Tasks:**
- [ ] Implement the logic grouping inference windows into 15m–90m "Session Aggregates".
- [ ] Apply the `60% dominant mode` aggregation rule (otherwise resolving to `Mixed`).
- [ ] Evaluate and map alignment of optional axes during aggregation.

**Codebase Touchpoints:**
- `src/taskclf/infer/batch.py`: Contains smoothing and merging logic (`rolling_majority`, `segmentize`, `merge_short_segments`) which handles grouping contiguous chunks. This will need to be refactored to implement the 60% dominance rule.
- `src/taskclf/core/types.py`: `LabelSpan` defines contiguous time spans for labels. This may need updating or extending to support `SessionMode` aggregates.

---

## Phase 6: Playbook & Migration
*Tying it together and back-porting existing data.*

**Tasks:**
- [ ] Draft the "Annotation playbook" containing canonical/boundary examples and the dominance checklist for the team.
- [ ] Write a data migration/mapping script to convert `v1` datasets to the `labels_v2` envelope, ensuring backward compatibility.

**Codebase Touchpoints:**
- `docs/guide/labels_v1.md`: Create a new `labels_v2_playbook.md` to serve as the annotation guide.
- New scripts/files needed in `src/taskclf/data/` or `src/taskclf/cli/` to migrate existing datasets containing `LABEL_SET_V1` strings into `LabelEnvelope` structured records.
