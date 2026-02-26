"""Model registry: scan, validate, rank, and filter model bundles.

Provides a pure, testable API for discovering promoted model bundles
under ``models/``, checking compatibility with the current schema and
label set, and ranking candidates by the selection policy.

Public surface:

* :class:`BundleMetrics` — parsed ``metrics.json``
* :class:`ModelBundle` — one scanned bundle (valid or invalid)
* :class:`SelectionPolicy` — ranking / constraint configuration
* :class:`ExclusionRecord` — why a bundle was excluded from selection
* :class:`SelectionReport` — full result of :func:`find_best_model`
* :func:`list_bundles` — scan a directory for bundles
* :func:`is_compatible` — schema hash + label set gate
* :func:`passes_constraints` — hard constraint gate (policy v1: no-op)
* :func:`score` — sortable ranking tuple
* :func:`find_best_model` — scan, filter, rank, and select the best bundle
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel

from taskclf.core.model_io import ModelMetadata
from taskclf.core.schema import FeatureSchemaV1
from taskclf.core.types import LABEL_SET_V1

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class BundleMetrics(BaseModel, frozen=True):
    """Metrics stored in a bundle's ``metrics.json``.

    See ``docs/guide/metrics_contract.md`` for the stable contract.
    """

    macro_f1: float
    weighted_f1: float
    confusion_matrix: list[list[int]]
    label_names: list[str]


class ModelBundle(BaseModel, frozen=True):
    """A scanned model bundle directory.

    Both valid and invalid bundles are represented; check :attr:`valid`
    before using :attr:`metadata` or :attr:`metrics`.
    """

    model_id: str
    path: Path
    valid: bool
    invalid_reason: str | None = None
    metadata: ModelMetadata | None = None
    metrics: BundleMetrics | None = None
    created_at: datetime | None = None


class SelectionPolicy(BaseModel, frozen=True):
    """Selection policy configuration.

    Policy v1 ranks by ``macro_f1`` desc, ``weighted_f1`` desc,
    ``created_at`` desc and applies no additional hard constraints
    (acceptance gates are enforced at promotion time by retrain).
    """

    version: int = 1


class ExclusionRecord(BaseModel, frozen=True):
    """Why a single bundle was excluded during :func:`find_best_model`."""

    model_id: str
    path: Path
    reason: str


class SelectionReport(BaseModel, frozen=True):
    """Full result of :func:`find_best_model`.

    *ranked* contains eligible bundles in score-descending order.
    *best* is ``ranked[0]`` when the list is non-empty, else ``None``.
    *excluded* lists every bundle that was filtered out, with a
    human-readable reason.
    """

    best: ModelBundle | None
    ranked: list[ModelBundle]
    excluded: list[ExclusionRecord]
    policy: SelectionPolicy
    required_schema_hash: str


# ---------------------------------------------------------------------------
# Bundle scanning
# ---------------------------------------------------------------------------


def _parse_bundle(bundle_dir: Path) -> ModelBundle:
    """Attempt to parse a single bundle directory into a :class:`ModelBundle`."""
    model_id = bundle_dir.name

    meta_path = bundle_dir / "metadata.json"
    if not meta_path.is_file():
        return ModelBundle(
            model_id=model_id,
            path=bundle_dir,
            valid=False,
            invalid_reason="missing metadata.json",
        )

    metrics_path = bundle_dir / "metrics.json"
    if not metrics_path.is_file():
        return ModelBundle(
            model_id=model_id,
            path=bundle_dir,
            valid=False,
            invalid_reason="missing metrics.json",
        )

    try:
        meta_raw = json.loads(meta_path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        return ModelBundle(
            model_id=model_id,
            path=bundle_dir,
            valid=False,
            invalid_reason=f"metadata.json parse error: {exc}",
        )

    try:
        metadata = ModelMetadata.model_validate(meta_raw)
    except Exception as exc:
        return ModelBundle(
            model_id=model_id,
            path=bundle_dir,
            valid=False,
            invalid_reason=f"metadata.json validation error: {exc}",
        )

    try:
        metrics_raw = json.loads(metrics_path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        return ModelBundle(
            model_id=model_id,
            path=bundle_dir,
            valid=False,
            invalid_reason=f"metrics.json parse error: {exc}",
        )

    try:
        metrics = BundleMetrics.model_validate(metrics_raw)
    except Exception as exc:
        return ModelBundle(
            model_id=model_id,
            path=bundle_dir,
            valid=False,
            invalid_reason=f"metrics.json validation error: {exc}",
        )

    try:
        created_at = datetime.fromisoformat(metadata.created_at)
    except (ValueError, TypeError) as exc:
        return ModelBundle(
            model_id=model_id,
            path=bundle_dir,
            valid=False,
            invalid_reason=f"created_at parse error: {exc}",
        )

    return ModelBundle(
        model_id=model_id,
        path=bundle_dir,
        valid=True,
        metadata=metadata,
        metrics=metrics,
        created_at=created_at,
    )


def list_bundles(models_dir: Path) -> list[ModelBundle]:
    """Scan *models_dir* for model bundle subdirectories.

    Each subdirectory is parsed independently; failures are captured as
    invalid bundles rather than aborting the scan.

    Args:
        models_dir: Parent directory containing bundle subdirectories
            (e.g. ``Path("models")``).

    Returns:
        A list of :class:`ModelBundle` instances sorted by ``model_id``
        for deterministic ordering.
    """
    if not models_dir.is_dir():
        return []

    bundles: list[ModelBundle] = []
    for candidate in models_dir.iterdir():
        if not candidate.is_dir():
            continue
        bundle = _parse_bundle(candidate)
        bundles.append(bundle)

    bundles.sort(key=lambda b: b.model_id)
    return bundles


# ---------------------------------------------------------------------------
# Compatibility & constraints
# ---------------------------------------------------------------------------


def is_compatible(
    bundle: ModelBundle,
    required_schema_hash: str = FeatureSchemaV1.SCHEMA_HASH,
    required_label_set: frozenset[str] = LABEL_SET_V1,
) -> bool:
    """Check whether *bundle* is compatible with the current runtime.

    A bundle is compatible when both hold:

    1. ``metadata.schema_hash`` exactly matches *required_schema_hash*.
    2. ``sorted(metadata.label_set)`` exactly matches ``sorted(required_label_set)``.

    Args:
        bundle: A scanned model bundle.
        required_schema_hash: Expected schema hash (defaults to
            ``FeatureSchemaV1.SCHEMA_HASH``).
        required_label_set: Expected label vocabulary (defaults to
            ``LABEL_SET_V1``).

    Returns:
        ``True`` if the bundle is valid and compatible.
    """
    if not bundle.valid or bundle.metadata is None:
        return False
    if bundle.metadata.schema_hash != required_schema_hash:
        return False
    if sorted(bundle.metadata.label_set) != sorted(required_label_set):
        return False
    return True


def passes_constraints(
    bundle: ModelBundle,
    policy: SelectionPolicy,  # noqa: ARG001 — reserved for future policy versions
) -> bool:
    """Check whether *bundle* passes the hard constraints of *policy*.

    Policy v1 applies no additional constraints beyond validity: all
    acceptance gates are enforced at promotion time by retrain, so any
    promoted bundle that parsed successfully is eligible for ranking.

    Args:
        bundle: A scanned model bundle (must be valid).
        policy: Selection policy configuration.

    Returns:
        ``True`` if the bundle is valid with non-null metrics.
    """
    return bundle.valid and bundle.metrics is not None


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------


def score(
    bundle: ModelBundle,
    policy: SelectionPolicy,  # noqa: ARG001 — reserved for future policy versions
) -> tuple[float, float, str]:
    """Compute a sortable ranking key for *bundle*.

    The tuple sorts **descending** on all three components:

    1. ``macro_f1`` (higher is better)
    2. ``weighted_f1`` (tie-break; higher is better)
    3. ``created_at`` (tie-break; newer is better — ISO8601 strings
       with the same UTC offset sort lexicographically)

    Args:
        bundle: A valid model bundle with non-null metrics and metadata.
        policy: Selection policy configuration.

    Returns:
        A 3-tuple ``(macro_f1, weighted_f1, created_at_raw)`` suitable
        for ``sorted(..., reverse=True)``.

    Raises:
        ValueError: If the bundle is invalid or missing metrics/metadata.
    """
    if not bundle.valid or bundle.metrics is None or bundle.metadata is None:
        raise ValueError(
            f"Cannot score invalid bundle {bundle.model_id!r}"
        )
    return (
        bundle.metrics.macro_f1,
        bundle.metrics.weighted_f1,
        bundle.metadata.created_at,
    )


# ---------------------------------------------------------------------------
# Best-model selection
# ---------------------------------------------------------------------------


def find_best_model(
    models_dir: Path,
    policy: SelectionPolicy | None = None,
    required_schema_hash: str | None = None,
    required_label_set: frozenset[str] | None = None,
) -> SelectionReport:
    """Scan, filter, rank, and select the best model bundle.

    This is the main entry-point for non-mutating model selection.
    It composes :func:`list_bundles`, :func:`is_compatible`,
    :func:`passes_constraints`, and :func:`score` into a single call
    that returns a structured :class:`SelectionReport`.

    Args:
        models_dir: Directory containing promoted model bundle
            subdirectories (e.g. ``Path("models")``).
        policy: Selection policy configuration.  Defaults to
            ``SelectionPolicy()`` (policy v1).
        required_schema_hash: Schema hash that bundles must match.
            Defaults to ``FeatureSchemaV1.SCHEMA_HASH``.
        required_label_set: Label vocabulary that bundles must match.
            Defaults to ``LABEL_SET_V1``.

    Returns:
        A :class:`SelectionReport` with the best bundle (if any),
        the full ranked list of eligible bundles, and exclusion
        records for every bundle that was filtered out.
    """
    if policy is None:
        policy = SelectionPolicy()
    if required_schema_hash is None:
        required_schema_hash = FeatureSchemaV1.SCHEMA_HASH
    if required_label_set is None:
        required_label_set = LABEL_SET_V1

    bundles = list_bundles(models_dir)

    excluded: list[ExclusionRecord] = []
    eligible: list[ModelBundle] = []

    for bundle in bundles:
        if not bundle.valid:
            excluded.append(ExclusionRecord(
                model_id=bundle.model_id,
                path=bundle.path,
                reason=f"invalid: {bundle.invalid_reason}",
            ))
            continue

        if not is_compatible(bundle, required_schema_hash, required_label_set):
            assert bundle.metadata is not None
            if bundle.metadata.schema_hash != required_schema_hash:
                detail = "schema_hash mismatch"
            else:
                detail = "label_set mismatch"
            excluded.append(ExclusionRecord(
                model_id=bundle.model_id,
                path=bundle.path,
                reason=f"incompatible: {detail}",
            ))
            continue

        if not passes_constraints(bundle, policy):
            excluded.append(ExclusionRecord(
                model_id=bundle.model_id,
                path=bundle.path,
                reason="constraint: failed policy constraints",
            ))
            continue

        eligible.append(bundle)

    ranked = sorted(
        eligible,
        key=lambda b: score(b, policy),
        reverse=True,
    )

    return SelectionReport(
        best=ranked[0] if ranked else None,
        ranked=ranked,
        excluded=excluded,
        policy=policy,
        required_schema_hash=required_schema_hash,
    )
