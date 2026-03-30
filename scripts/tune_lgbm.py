"""Hyperparameter tuning for LightGBM via exhaustive grid search.

Run from the repo root:

    uv run python scripts/tune_lgbm.py

Explores the parameter grid defined in Step 7.1 of TODO_INFERENCE_QUALITY.md,
evaluates every combination on a time-based validation split, and writes a
results report to artifacts/experiments/hyperparameter_tuning/.
"""

from __future__ import annotations

import datetime as dt
import itertools
import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

from taskclf.core.defaults import DEFAULT_MODELS_DIR
from taskclf.core.model_io import load_model_bundle
from taskclf.features.build import generate_dummy_features
from taskclf.labels.projection import project_blocks_to_windows
from taskclf.labels.store import generate_dummy_labels
from taskclf.train.dataset import split_by_time
from taskclf.train.evaluate import evaluate_model
from taskclf.train.lgbm import train_lgbm
from taskclf.train.retrain import (
    RetrainConfig,
    check_candidate_gates,
    check_regression_gates,
    find_latest_model,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)


PARAM_GRID: dict[str, list[Any]] = {
    "num_leaves": [15, 31, 63, 127],
    "min_data_in_leaf": [5, 10, 20, 50],
    "feature_fraction": [0.6, 0.8, 1.0],
    "bagging_fraction": [0.6, 0.8, 1.0],
    "lambda_l1": [0, 0.1, 1.0],
    "lambda_l2": [0, 0.1, 1.0],
}

LR_ROUNDS_PAIRS: list[tuple[float, int]] = [
    (0.1, 200),
    (0.05, 400),
    (0.01, 1000),
]


def _build_labeled_data(
    n_days: int = 10,
    n_rows_per_day: int = 60,
    n_users: int = 2,
) -> pd.DataFrame:
    """Build a synthetic labeled DataFrame large enough for meaningful tuning."""
    users = [f"tune-user-{i:03d}" for i in range(n_users)]
    all_feats: list[pd.DataFrame] = []
    all_labels = []

    for uid in users:
        for d_offset in range(n_days):
            date = dt.date(2025, 7, 1) + dt.timedelta(days=d_offset)
            rows = generate_dummy_features(date, n_rows=n_rows_per_day, user_id=uid)
            df = pd.DataFrame([r.model_dump() for r in rows])
            labels = generate_dummy_labels(date, n_rows=n_rows_per_day)
            all_feats.append(df)
            all_labels.extend(labels)

    features_df = pd.concat(all_feats, ignore_index=True)
    labeled_df = project_blocks_to_windows(features_df, all_labels)
    return labeled_df


def _resolve_champion(
    models_dir: Path,
) -> tuple[Any, dict[str, Any] | None] | None:
    """Load the current champion model and its categorical encoders, if any."""
    champion_path = find_latest_model(models_dir)
    if champion_path is None:
        return None
    try:
        model, _meta, cat_encoders = load_model_bundle(champion_path)
        return model, cat_encoders
    except (ValueError, OSError) as exc:
        logger.warning("Could not load champion model: %s", exc)
        return None


def run_grid_search() -> dict[str, Any]:
    """Execute the full hyperparameter grid search and return a results dict."""
    logger.info("Building synthetic labeled data …")
    labeled_df = _build_labeled_data()
    logger.info("Labeled dataset: %d rows", len(labeled_df))

    if labeled_df.empty:
        logger.error("No labeled rows — cannot tune")
        sys.exit(1)

    splits = split_by_time(labeled_df)
    train_df = labeled_df.iloc[splits["train"]].reset_index(drop=True)
    val_df = labeled_df.iloc[splits["val"]].reset_index(drop=True)
    test_df = labeled_df.iloc[splits["test"]].reset_index(drop=True)

    if train_df.empty or val_df.empty:
        logger.error("Train or val split is empty — need more data")
        sys.exit(1)

    logger.info(
        "Split sizes: train=%d  val=%d  test=%d",
        len(train_df),
        len(val_df),
        len(test_df),
    )

    grid_keys = sorted(PARAM_GRID.keys())
    grid_values = [PARAM_GRID[k] for k in grid_keys]
    static_combos = list(itertools.product(*grid_values))
    all_combos = [
        (dict(zip(grid_keys, combo)), lr, rounds)
        for combo in static_combos
        for lr, rounds in LR_ROUNDS_PAIRS
    ]
    total_possible = len(all_combos)

    max_trials = min(200, total_possible)
    rng = random.Random(42)
    sampled = (
        rng.sample(all_combos, max_trials)
        if max_trials < total_possible
        else all_combos
    )
    logger.info(
        "Grid: %d possible combinations, sampling %d trials",
        total_possible,
        len(sampled),
    )

    best_macro_f1 = -1.0
    best_params: dict[str, Any] = {}
    best_num_boost_round = 100
    all_results: list[dict[str, Any]] = []

    for trial_idx, (extra, lr, rounds) in enumerate(sampled):
        trial_params = {**extra, "learning_rate": lr}

        if trial_idx % 50 == 0:
            logger.info("Trial %d / %d …", trial_idx + 1, len(sampled))

        t0 = time.monotonic()
        try:
            _model, metrics, _cm, _params, cat_encoders = train_lgbm(
                train_df,
                val_df,
                num_boost_round=rounds,
                extra_params=trial_params,
            )
        except Exception:
            logger.debug("Trial %d failed", trial_idx + 1, exc_info=True)
            continue
        elapsed = time.monotonic() - t0

        macro_f1 = metrics["macro_f1"]

        record = {
            **trial_params,
            "num_boost_round": rounds,
            "macro_f1": round(macro_f1, 4),
            "weighted_f1": round(metrics["weighted_f1"], 4),
            "elapsed_s": round(elapsed, 2),
        }
        all_results.append(record)

        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_params = dict(trial_params)
            best_num_boost_round = rounds

    logger.info(
        "Best macro_f1=%.4f with params=%s  num_boost_round=%d",
        best_macro_f1,
        best_params,
        best_num_boost_round,
    )

    # Retrain best model and evaluate on test set
    logger.info("Retraining best model on full train set …")
    best_model, best_metrics, best_cm, _, best_cat_encoders = train_lgbm(
        train_df,
        val_df,
        num_boost_round=best_num_boost_round,
        extra_params=best_params,
    )

    eval_df = test_df if not test_df.empty else val_df
    challenger_report = evaluate_model(
        best_model, eval_df, cat_encoders=best_cat_encoders
    )

    candidate_gates = check_candidate_gates(challenger_report)
    logger.info("Candidate gates: all_passed=%s", candidate_gates.all_passed)
    for g in candidate_gates.gates:
        logger.info("  %s: %s — %s", g.name, g.passed, g.detail)

    # Compare against champion if one exists
    models_dir = Path(DEFAULT_MODELS_DIR)
    champion_result = _resolve_champion(models_dir)
    champion_comparison: dict[str, Any] | None = None

    if champion_result is not None:
        champ_model, champ_encoders = champion_result
        champion_report = evaluate_model(
            champ_model, eval_df, cat_encoders=champ_encoders
        )
        config = RetrainConfig()
        regression = check_regression_gates(champion_report, challenger_report, config)

        champion_comparison = {
            "champion_macro_f1": round(champion_report.macro_f1, 4),
            "challenger_macro_f1": round(challenger_report.macro_f1, 4),
            "regression_all_passed": regression.all_passed,
            "gates": [
                {"name": g.name, "passed": g.passed, "detail": g.detail}
                for g in regression.gates
            ],
        }
        logger.info(
            "Champion comparison: champion=%.4f  challenger=%.4f  passed=%s",
            champion_report.macro_f1,
            challenger_report.macro_f1,
            regression.all_passed,
        )
    else:
        logger.info("No champion model found — skipping regression comparison")

    return {
        "best_params": {**best_params, "num_boost_round": best_num_boost_round},
        "best_macro_f1": round(best_macro_f1, 4),
        "challenger_test_report": {
            "macro_f1": round(challenger_report.macro_f1, 4),
            "weighted_f1": round(challenger_report.weighted_f1, 4),
            "per_class": challenger_report.per_class,
            "acceptance_checks": challenger_report.acceptance_checks,
            "acceptance_details": challenger_report.acceptance_details,
            "reject_rate": challenger_report.reject_rate,
        },
        "candidate_gates": {
            "all_passed": candidate_gates.all_passed,
            "gates": [
                {"name": g.name, "passed": g.passed, "detail": g.detail}
                for g in candidate_gates.gates
            ],
        },
        "champion_comparison": champion_comparison,
        "total_possible": total_possible,
        "sampled_trials": len(sampled),
        "completed_trials": len(all_results),
        "all_trials": sorted(all_results, key=lambda r: r["macro_f1"], reverse=True),
        "data_info": {
            "labeled_rows": len(labeled_df),
            "train_rows": len(train_df),
            "val_rows": len(val_df),
            "test_rows": len(test_df),
        },
    }


def write_results_markdown(results: dict[str, Any], output_dir: Path) -> Path:
    """Write a human-readable results report as markdown."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "results.md"

    bp = results["best_params"]
    tr = results["challenger_test_report"]
    top_trials = results["all_trials"][:20]

    lines = [
        "# Hyperparameter Tuning Results",
        "",
        f"**Date**: {dt.date.today().isoformat()}",
        f"**Grid size**: {results['total_possible']}",
        f"**Sampled trials**: {results['sampled_trials']}",
        f"**Completed trials**: {results['completed_trials']}",
        "",
        "## Data",
        "",
        f"- Labeled rows: {results['data_info']['labeled_rows']}",
        f"- Train: {results['data_info']['train_rows']}",
        f"- Val: {results['data_info']['val_rows']}",
        f"- Test: {results['data_info']['test_rows']}",
        "- Split method: `split_by_time()` (chronological per-user)",
        "",
        "## Best Parameters",
        "",
        f"- **Validation macro-F1**: {results['best_macro_f1']}",
        "",
        "```json",
        json.dumps(bp, indent=2),
        "```",
        "",
        "## Test-Set Evaluation (best model)",
        "",
        f"- **macro-F1**: {tr['macro_f1']}",
        f"- **weighted-F1**: {tr['weighted_f1']}",
        f"- **reject rate**: {tr['reject_rate']}",
        "",
        "### Acceptance checks",
        "",
    ]

    for check, passed in tr["acceptance_checks"].items():
        status = "PASS" if passed else "FAIL"
        detail = tr["acceptance_details"].get(check, "")
        lines.append(f"- **{check}**: {status} — {detail}")

    lines.extend(["", "### Per-class metrics", ""])
    lines.append("| Class | Precision | Recall | F1 |")
    lines.append("|-------|-----------|--------|----|")
    for cls, m in sorted(tr["per_class"].items()):
        lines.append(
            f"| {cls} | {m.get('precision', 0):.4f} | "
            f"{m.get('recall', 0):.4f} | {m.get('f1-score', 0):.4f} |"
        )

    cg = results["candidate_gates"]
    lines.extend(
        [
            "",
            "## Candidate Gates",
            "",
            f"**All passed**: {cg['all_passed']}",
            "",
        ]
    )
    for g in cg["gates"]:
        status = "PASS" if g["passed"] else "FAIL"
        lines.append(f"- **{g['name']}**: {status} — {g['detail']}")

    cc = results["champion_comparison"]
    if cc is not None:
        lines.extend(
            [
                "",
                "## Champion Comparison",
                "",
                f"- **Champion macro-F1**: {cc['champion_macro_f1']}",
                f"- **Challenger macro-F1**: {cc['challenger_macro_f1']}",
                f"- **Regression gates passed**: {cc['regression_all_passed']}",
                "",
            ]
        )
        for g in cc["gates"]:
            status = "PASS" if g["passed"] else "FAIL"
            lines.append(f"- **{g['name']}**: {status} — {g['detail']}")
    else:
        lines.extend(
            [
                "",
                "## Champion Comparison",
                "",
                "No champion model found. Skipped regression comparison.",
            ]
        )

    lines.extend(
        [
            "",
            "## Top 20 Trials (by validation macro-F1)",
            "",
            "| # | num_leaves | min_data_in_leaf | feature_fraction | "
            "bagging_fraction | lambda_l1 | lambda_l2 | learning_rate | "
            "num_boost_round | macro_f1 | weighted_f1 | elapsed_s |",
            "|---|------------|------------------|------------------|"
            "-----------------|-----------|-----------|---------------|"
            "-----------------|----------|-------------|-----------|",
        ]
    )
    for i, t in enumerate(top_trials, 1):
        lines.append(
            f"| {i} | {t.get('num_leaves', '')} | "
            f"{t.get('min_data_in_leaf', '')} | "
            f"{t.get('feature_fraction', '')} | "
            f"{t.get('bagging_fraction', '')} | "
            f"{t.get('lambda_l1', '')} | "
            f"{t.get('lambda_l2', '')} | "
            f"{t.get('learning_rate', '')} | "
            f"{t.get('num_boost_round', '')} | "
            f"{t['macro_f1']} | {t['weighted_f1']} | {t['elapsed_s']} |"
        )

    lines.append("")
    out_path.write_text("\n".join(lines))
    return out_path


def main() -> None:
    results = run_grid_search()

    output_dir = Path("artifacts/experiments/hyperparameter_tuning")
    md_path = write_results_markdown(results, output_dir)
    logger.info("Results written to %s", md_path)

    json_path = output_dir / "results.json"
    json_path.write_text(json.dumps(results, indent=2, default=str))
    logger.info("Raw JSON results written to %s", json_path)

    bp = results["best_params"]
    logger.info(
        "\nBest parameters found:\n%s\nValidation macro-F1: %.4f\nTest macro-F1: %.4f",
        json.dumps(bp, indent=2),
        results["best_macro_f1"],
        results["challenger_test_report"]["macro_f1"],
    )

    cc = results["champion_comparison"]
    if cc is not None:
        if cc["regression_all_passed"] and (
            cc["challenger_macro_f1"] > cc["champion_macro_f1"]
        ):
            logger.info(
                "Tuned model is STRICTLY BETTER than champion. "
                "Consider updating _DEFAULT_PARAMS in lgbm.py."
            )
        elif cc["regression_all_passed"]:
            logger.info(
                "Tuned model passes regression gates but is not strictly better."
            )
        else:
            logger.info("Tuned model FAILS regression gates. Do not update defaults.")


if __name__ == "__main__":
    main()
