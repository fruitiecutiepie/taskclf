## TODOs (ordered)

### 0) Lock down invariants and contracts (context doc)

1. ~~**Create `docs/model_selection.md`** describing:~~

   * ~~Definitions: *bundle*, *promoted model*, *eligible model*, *active model*, *override model*.~~
   * ~~Selection policy v1:~~

     * ~~Hard filters (must pass): schema hash match, acceptance_pass, BreakIdle precision ≥ 0.95, min_class_precision ≥ 0.50 (or whatever your canonical gates are).~~
     * ~~Ranking metric: macro-F1 desc; tie-break weighted-F1 desc; tie-break created_at desc.~~
   * ~~“Default model” contract for inference:~~

     * ~~If `--model-dir` provided → use it.~~
     * ~~Else → use `models/active.json` if present, otherwise compute selection and optionally write it.~~
   * ~~Atomicity + audit expectations:~~

     * ~~`active.json` written atomically; append `active_history.jsonl`.~~
2. ~~**Create `docs/metrics_contract.md`** (or update existing docs) with exact JSON keys/types used by selection:~~

   * ~~Required: `macro_f1`, `weighted_f1`, `acceptance_pass` (bool), `breakidle_precision`, `min_class_precision`~~
   * ~~Optional: confusion matrix, per-class metrics, eval_dataset_id, created_at.~~
3. ~~**Create `docs/model_bundle_layout.md`**:~~

   * ~~Required files per bundle: `metadata.json`, `metrics.json`, model artifact(s).~~
   * ~~Required metadata keys: `model_id`, `created_at`, `schema_hash`, `train_range`, `params_hash` (as applicable).~~

> Agent injection: give agents `docs/model_selection.md` + `docs/metrics_contract.md` + a sample of an existing `metrics.json`/`metadata.json`.

---

### ~~1) Implement the model registry scanner (pure, testable)~~

4. ~~**Add module `src/taskclf/model_registry.py`** with:~~

   * ~~`list_bundles(models_dir) -> list[ModelBundle]` reading `metrics.json` + `metadata.json`.~~
   * ~~Strict validation: missing required keys => mark bundle invalid with reason (don’t crash whole scan).~~
   * ~~Deterministic ordering and stable parsing of timestamps (store raw + parsed).~~
5. ~~**Add `ModelBundle` dataclass** with:~~

   * ~~`model_id`, `path`, `created_at`, `schema_hash`, `metrics`, `metadata`.~~
6. ~~**Add `is_compatible(bundle, required_schema_hash)`**.~~
7. ~~**Add `passes_constraints(bundle, policy)`** implementing your regression gates as hard constraints (not “macro-f1 no regression” yet; that needs a baseline).~~
8. ~~**Add `score(bundle, policy)`** returning a sortable tuple for ranking.~~

> ~~Context needed: where “current schema hash” is sourced (CLI flag? config file? computed from feature schema). Document it in `docs/model_selection.md`.~~

---

### ~~2) Implement best-model selection (non-mutating)~~

9. ~~**Implement `find_best_model(models_dir, policy, required_schema_hash=None)`**~~

   * ~~Scans `models/`~~
   * ~~Filters compatible + passes_constraints~~
   * ~~Ranks by score~~
   * ~~Returns best bundle + a structured `SelectionReport` (why others were excluded, ranking list).~~
10. ~~**Add unit tests** for:~~

* ~~Missing files/keys handled gracefully~~
* ~~Schema mismatch excluded~~
* ~~Constraint failure excluded~~
* ~~Ranking/tie-breaks deterministic~~

> ~~Agent injection: include a tiny fixture set of bundles under `tests/fixtures/models/` (3–6 bundles) with deliberate edge cases.~~

---

### ~~3) Add “active model pointer” (recommended) + atomic writer~~

11. ~~**Add `models/active.json` concept** (even if you also support “best by scan”):~~

* ~~`read_active(models_dir) -> ActivePointer|None`~~
* ~~`write_active_atomic(models_dir, bundle, policy, reason)`~~
* ~~`append_active_history(models_dir, old, new)`~~

12. ~~**Define `ActivePointer` schema** in `docs/model_selection.md` and validate on read.~~
13. ~~**Add a guard**: if `active.json` points to missing path, fall back to selection and repair (optional).~~

> Why: “continuously” using best model is easier if training updates a pointer; inference shouldn’t rescan each time.

---

### ~~4) Wire selection into retrain/promotion (the overthrow mechanism)~~

14. ~~**Replace/augment `find_latest_model()` usage** in retrain:~~

* ~~Use `read_active()` as the champion baseline (if exists), else fallback to `find_best_model()` or latest promoted.~~

15. ~~**After promotion to `models/<id>/`**, run:~~

* ~~`best = find_best_model(...)`~~
* ~~If `best != active` → `write_active_atomic(...)`~~

16. ~~**Update `check_regression_gates()` inputs** to clearly distinguish:~~

* ~~“Candidate passes hard gates” (candidate-only)~~
* ~~“Candidate beats champion” (comparative). Keep comparative logic in retrain only.~~

> ~~Context needed: where “promoted model bundles” live and how model IDs are formed.~~

---

### ~~5) Inference defaulting (CLI + library API)~~

17. ~~**Define a single resolver** `resolve_model_dir(args, models_dir, policy)`:~~

* If `--model-dir` given → return it
* Else:

  * If `active.json` exists and valid → return its path
  * Else → compute `best` and return its path (optionally write `active.json` if you want self-healing)

18. ~~**Update batch inference CLI** to make `--model-dir` optional.~~
19. ~~**Update online inference**:~~

* Add lightweight reload mechanism:

  * Poll `active.json` mtime every N seconds OR reload per request with caching.
  * Swap model only after new model loads successfully.

20. ~~**Add integration tests**:~~

* “No model-dir and active exists” uses active
* “No active and models exist” picks best
* “Active points to missing” falls back

> Context needed: how inference loads LightGBM (file name, booster serialization) and where it expects artifacts.

---

### ~~6) Add `taskclf train list` (ranked view)~~

21. ~~**Implement CLI command** `taskclf train list`:~~

* ~~Outputs table: model_id, created_at, schema_hash, macro_f1, weighted_f1, breakidle_precision, min_class_precision, acceptance_pass, promoted(bool), active(bool)~~
* ~~Supports `--sort macro_f1|weighted_f1|created_at`, `--all/--eligible`, `--schema-hash <...>`~~

22. ~~**Add output formats**:~~

* ~~default pretty table~~
* ~~`--json` for automation~~

---

### ~~7) Performance + operational hardening (optional but sensible)~~

23. ~~**Cache scan results** (optional):~~

* ~~Write `models/index.json` with computed ranking + invalid reasons; refresh on retrain/promotion.~~

24. ~~**Add hysteresis** (optional):~~

* ~~`--min-improvement 0.002` macro-F1 improvement needed to switch active.~~

25. ~~**Add rollback command**:~~

* ~~`taskclf model set-active --model-id ...` (writes `active.json` atomically + logs)~~
