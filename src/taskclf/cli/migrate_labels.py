#!/usr/bin/env python3
"""Migrate v1 label datasets to the labels_v2 Envelope format.

Reads existing CSV files containing `LABEL_SET_V1` strings and emits JSONL
records conforming to the `LabelEnvelope` schema, maintaining backward
compatibility through a deterministic mapping.

Usage:
    python -m taskclf.cli.migrate_labels <input_v1.csv> <output_v2.jsonl>
"""

import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from taskclf.core.types import (
    AxisDecision,
    IntentBasis,
    LabelEnvelope,
    Mode,
    ModeSource,
    SemanticLabel,
    Subtype,
    SupportState,
)

logger = logging.getLogger(__name__)

# --- V1 to V2 Migration Mapping ---
# Follows the recommended fallback mappings from labels_v2 design.
_V1_MAPPING: dict[str, tuple[Mode, Subtype | None]] = {
    "Build": (Mode.Produce, Subtype.Build),
    "Debug": (Mode.Produce, Subtype.Debug),
    "Write": (Mode.Produce, Subtype.Write),
    "Review": (Mode.Produce, Subtype.Review),  # Defaulting Review to Produce
    "ReadResearch": (Mode.Consume, Subtype.ReadResearch),
    "Communicate": (Mode.Coordinate, Subtype.Communicate),
    "Meet": (Mode.Attend, Subtype.Meet),
    "BreakIdle": (Mode.Idle, Subtype.BreakIdle),
    "Mixed/Unknown": (Mode.Idle, None),  # Catch-all
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Migrate v1 labels to v2 envelopes.")
    parser.add_argument("input_csv", type=Path, help="Path to input v1 labels CSV.")
    parser.add_argument("output_jsonl", type=Path, help="Path to output v2 JSONL.")
    parser.add_argument(
        "--rule-version",
        default=f"{datetime.now(timezone.utc).strftime('%Y-%m-%d')}.1",
        help="Rule version to stamp on the envelopes.",
    )
    return parser.parse_args()


def migrate_row(row: pd.Series, rule_version: str) -> LabelEnvelope:
    """Convert a single v1 label row to a v2 LabelEnvelope."""
    v1_label = str(row.get("label", "Mixed/Unknown"))
    start_ts_str = str(row.get("start_ts", ""))
    end_ts_str = str(row.get("end_ts", ""))
    provenance = str(row.get("provenance", "")).strip().lower()

    # Calculate inference window MS if possible
    inference_ms = 180000  # Default 3 minutes
    try:
        if start_ts_str and end_ts_str:
            start_ts = datetime.fromisoformat(start_ts_str.replace("Z", "+00:00"))
            end_ts = datetime.fromisoformat(end_ts_str.replace("Z", "+00:00"))
            inference_ms = int((end_ts - start_ts).total_seconds() * 1000)
    except Exception:
        pass

    confidence = row.get("confidence", 1.0)
    if pd.isna(confidence):
        confidence = 1.0

    if v1_label == "Mixed/Unknown":
        confidence = 0.3  # Do not force false confidence on ambiguous legacy labels

    mode, subtype = _V1_MAPPING.get(v1_label, (Mode.Idle, None))

    support_state = SupportState.Supported
    if v1_label == "Mixed/Unknown" or confidence < 0.5:
        support_state = SupportState.MixedUnknown
    elif confidence < 0.7:
        support_state = SupportState.WeakEvidence

    mode_decision = AxisDecision[Mode](
        value=mode, confidence=float(confidence), reason_codes=["v1_legacy_migration"]
    )

    subtype_decision = None
    if subtype:
        subtype_decision = AxisDecision[Subtype](
            value=subtype,
            confidence=float(confidence),
            reason_codes=["v1_legacy_migration"],
        )

    is_user_declared = any(
        token in provenance for token in ("manual", "user", "annotat")
    )
    intent_basis = IntentBasis.UserDeclared if is_user_declared else IntentBasis.Unknown
    mode_source = (
        ModeSource.UserOverride if is_user_declared else ModeSource.DeterministicRule
    )

    semantic_label = SemanticLabel(
        mode=mode_decision,
        subtype=subtype_decision,
        support_state=support_state,
        intent_basis=intent_basis,
        mode_source=mode_source,
        # Omit optional axes (interaction, collaboration, domain) for legacy data
    )

    return LabelEnvelope(
        taxonomy_version="labels_v2",
        rule_version=rule_version,
        generated_at=datetime.now(timezone.utc).isoformat(),
        evidence_window_ms=30000,  # Assumption of 30s
        inference_window_ms=inference_ms,
        observed=None,
        semantic=semantic_label,
        plugins=None,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()

    if not args.input_csv.exists():
        logger.error(f"Input file not found: {args.input_csv}")
        return

    logger.info(f"Reading v1 labels from {args.input_csv}")
    df = pd.read_csv(args.input_csv)

    # Require basic label structure
    if "label" not in df.columns:
        logger.error("Input CSV must contain a 'label' column.")
        return

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    success_count = 0
    with open(args.output_jsonl, "w") as f:
        for _, row in df.iterrows():
            try:
                envelope = migrate_row(row, args.rule_version)
                f.write(envelope.model_dump_json(exclude_none=True) + "\n")
                success_count += 1
            except Exception as e:
                logger.warning(f"Skipping malformed row: {e}")

    logger.info(
        f"Migration complete. Wrote {success_count} envelopes to {args.output_jsonl}"
    )


if __name__ == "__main__":
    main()
