"""Model registry: scan, validate, rank, filter, and activate model bundles.

Provides a pure, testable API for discovering promoted model bundles
under ``models/``, checking compatibility with the current schema and
label set, ranking candidates by the selection policy, and managing
the active model pointer.

Public surface:

* :class:`BundleMetrics` — parsed ``metrics.json``
* :class:`ModelBundle` — one scanned bundle (valid or invalid)
* :class:`SelectionPolicy` — ranking / constraint configuration
* :class:`ExclusionRecord` — why a bundle was excluded from selection
* :class:`SelectionReport` — full result of :func:`find_best_model`
* :class:`ActivePointer` — persisted ``active.json`` pointer
* :class:`ActiveHistoryEntry` — one line in ``active_history.jsonl``
* :class:`IndexCacheBundleSummary` — one bundle row in ``index.json``
* :class:`IndexCache` — cached scan/ranking snapshot
* :func:`list_bundles` — scan a directory for bundles
* :func:`is_compatible` — schema hash + label set gate
* :func:`passes_constraints` — hard constraint gate (policy v1: no-op)
* :func:`score` — sortable ranking tuple
* :func:`find_best_model` — scan, filter, rank, and select the best bundle
* :func:`read_active` — read ``active.json`` pointer
* :func:`write_active_atomic` — atomically update ``active.json``
* :func:`append_active_history` — append to ``active_history.jsonl``
* :func:`resolve_active_model` — resolve active bundle with fallback
* :func:`write_index_cache` — write ``index.json`` from a selection report
* :func:`read_index_cache` — read cached ``index.json``
* :func:`should_switch_active` — hysteresis check before switching active
"""

from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime
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

    ``min_improvement`` controls hysteresis: the candidate must exceed
    the current active model's ``macro_f1`` by at least this amount
    before the active pointer is switched.  Set to ``0.0`` (default)
    to disable hysteresis.
    """

    version: int = 1
    min_improvement: float = 0.0


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


class ActivePointer(BaseModel, frozen=True):
    """Persisted pointer to the currently active model bundle.

    Stored as ``models/active.json``.  See
    ``docs/guide/model_selection.md`` for the schema contract.
    """

    model_dir: str
    selected_at: str
    policy_version: int
    model_id: str | None = None
    reason: dict[str, object] | None = None


class ActiveHistoryEntry(BaseModel, frozen=True):
    """One line in ``models/active_history.jsonl``.

    Records every change to ``active.json`` for auditability and
    rollback.
    """

    at: str
    old: ActivePointer | None
    new: ActivePointer


class IndexCacheBundleSummary(BaseModel, frozen=True):
    """Summary of one bundle stored inside :class:`IndexCache`."""

    model_id: str
    path: str
    macro_f1: float | None = None
    weighted_f1: float | None = None
    created_at: str | None = None
    eligible: bool = False


class IndexCache(BaseModel, frozen=True):
    """Cached scan/ranking snapshot written to ``models/index.json``.

    This is an informational cache — selection never reads it.
    Operators and ``taskclf train list`` may consume it for fast
    inspection without a full rescan.
    """

    generated_at: str
    schema_hash: str
    policy_version: int
    ranked: list[IndexCacheBundleSummary]
    excluded: list[ExclusionRecord]
    best_model_id: str | None = None


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


# ---------------------------------------------------------------------------
# Active model pointer
# ---------------------------------------------------------------------------

_ACTIVE_FILE = "active.json"
_ACTIVE_TMP = "active.json.tmp"
_HISTORY_FILE = "active_history.jsonl"
_INDEX_FILE = "index.json"
_INDEX_TMP = "index.json.tmp"


def read_active(models_dir: Path) -> ActivePointer | None:
    """Read the active model pointer from ``models_dir/active.json``.

    Returns ``None`` (without raising) when the file is missing,
    contains invalid JSON, or fails :class:`ActivePointer` validation.
    A warning is logged on parse/validation failures so operators can
    notice stale pointer files.
    """
    active_path = models_dir / _ACTIVE_FILE
    if not active_path.is_file():
        return None

    try:
        raw = json.loads(active_path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("active.json parse error: %s", exc)
        return None

    try:
        return ActivePointer.model_validate(raw)
    except Exception as exc:
        logger.warning("active.json validation error: %s", exc)
        return None


def append_active_history(
    models_dir: Path,
    old: ActivePointer | None,
    new: ActivePointer,
) -> None:
    """Append a transition record to ``models_dir/active_history.jsonl``.

    Creates the file if it does not exist.  Each line is a
    self-contained JSON object matching :class:`ActiveHistoryEntry`.
    """
    entry = ActiveHistoryEntry(at=new.selected_at, old=old, new=new)
    history_path = models_dir / _HISTORY_FILE
    with history_path.open("a") as fh:
        fh.write(entry.model_dump_json() + "\n")


def write_active_atomic(
    models_dir: Path,
    bundle: ModelBundle,
    policy: SelectionPolicy,
    reason: str | None = None,
) -> ActivePointer:
    """Atomically write ``models_dir/active.json`` for *bundle*.

    The pointer is written to a temporary file first, then moved into
    place with :func:`os.replace` to guarantee readers never see a
    partial write.  The previous pointer (if any) is read before the
    overwrite and both old and new are appended to the audit log via
    :func:`append_active_history`.

    Args:
        models_dir: The ``models/`` directory.
        bundle: The bundle to activate (must be valid with metrics).
        policy: The selection policy used to choose this bundle.
        reason: Optional human-readable reason string.

    Returns:
        The newly written :class:`ActivePointer`.
    """
    old = read_active(models_dir)

    reason_dict: dict[str, object] | None = None
    if bundle.metrics is not None:
        reason_dict = {
            "metric": "macro_f1",
            "macro_f1": bundle.metrics.macro_f1,
            "weighted_f1": bundle.metrics.weighted_f1,
        }
        if reason is not None:
            reason_dict["note"] = reason

    model_dir = str(bundle.path.relative_to(models_dir.parent))

    pointer = ActivePointer(
        model_dir=model_dir,
        model_id=bundle.model_id,
        selected_at=datetime.now(UTC).isoformat(),
        policy_version=policy.version,
        reason=reason_dict,
    )

    tmp_path = models_dir / _ACTIVE_TMP
    final_path = models_dir / _ACTIVE_FILE

    tmp_path.write_text(pointer.model_dump_json(indent=2) + "\n")
    os.replace(tmp_path, final_path)

    append_active_history(models_dir, old, pointer)

    return pointer


def resolve_active_model(
    models_dir: Path,
    policy: SelectionPolicy | None = None,
    required_schema_hash: str | None = None,
    required_label_set: frozenset[str] | None = None,
) -> tuple[ModelBundle | None, SelectionReport | None]:
    """Resolve the active model bundle, falling back to selection.

    Resolution order:

    1. Read ``active.json``.  If valid and the pointed-to bundle
       exists, is parseable, and is compatible — return it immediately
       (no full scan).
    2. Otherwise fall back to :func:`find_best_model`.  If a best
       bundle is found, atomically update ``active.json`` to self-heal
       the pointer.

    Returns:
        A 2-tuple ``(bundle, report)``.  *report* is ``None`` when the
        pointer was valid and no scan was needed.
    """
    if policy is None:
        policy = SelectionPolicy()
    if required_schema_hash is None:
        required_schema_hash = FeatureSchemaV1.SCHEMA_HASH
    if required_label_set is None:
        required_label_set = LABEL_SET_V1

    pointer = read_active(models_dir)
    if pointer is not None:
        bundle_path = models_dir.parent / pointer.model_dir
        if bundle_path.is_dir():
            bundle = _parse_bundle(bundle_path)
            if bundle.valid and is_compatible(
                bundle, required_schema_hash, required_label_set
            ):
                return bundle, None
            logger.warning(
                "active.json points to invalid/incompatible bundle %s; "
                "falling back to selection",
                pointer.model_dir,
            )
        else:
            logger.warning(
                "active.json points to missing directory %s; "
                "falling back to selection",
                pointer.model_dir,
            )

    report = find_best_model(
        models_dir, policy, required_schema_hash, required_label_set,
    )

    if report.best is not None:
        write_active_atomic(models_dir, report.best, policy, reason="auto-repair")

    return report.best, report


# ---------------------------------------------------------------------------
# Index cache
# ---------------------------------------------------------------------------


def write_index_cache(
    models_dir: Path,
    report: SelectionReport,
) -> IndexCache:
    """Write ``models_dir/index.json`` from a :class:`SelectionReport`.

    The cache is written atomically (temp + :func:`os.replace`).  It is
    informational only — :func:`find_best_model` never reads it.

    Args:
        models_dir: The ``models/`` directory.
        report: A completed selection report.

    Returns:
        The :class:`IndexCache` that was persisted.
    """
    ranked_summaries: list[IndexCacheBundleSummary] = []
    for b in report.ranked:
        ranked_summaries.append(IndexCacheBundleSummary(
            model_id=b.model_id,
            path=str(b.path),
            macro_f1=b.metrics.macro_f1 if b.metrics else None,
            weighted_f1=b.metrics.weighted_f1 if b.metrics else None,
            created_at=b.metadata.created_at if b.metadata else None,
            eligible=True,
        ))

    cache = IndexCache(
        generated_at=datetime.now(UTC).isoformat(),
        schema_hash=report.required_schema_hash,
        policy_version=report.policy.version,
        ranked=ranked_summaries,
        excluded=report.excluded,
        best_model_id=report.best.model_id if report.best else None,
    )

    tmp_path = models_dir / _INDEX_TMP
    final_path = models_dir / _INDEX_FILE

    tmp_path.write_text(cache.model_dump_json(indent=2) + "\n")
    os.replace(tmp_path, final_path)

    return cache


def read_index_cache(models_dir: Path) -> IndexCache | None:
    """Read the cached index from ``models_dir/index.json``.

    Returns ``None`` (without raising) when the file is missing,
    contains invalid JSON, or fails :class:`IndexCache` validation.
    """
    index_path = models_dir / _INDEX_FILE
    if not index_path.is_file():
        return None

    try:
        raw = json.loads(index_path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("index.json parse error: %s", exc)
        return None

    try:
        return IndexCache.model_validate(raw)
    except Exception as exc:
        logger.warning("index.json validation error: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Hysteresis
# ---------------------------------------------------------------------------


def should_switch_active(
    current: ActivePointer | None,
    candidate: ModelBundle,
    policy: SelectionPolicy,
) -> bool:
    """Decide whether *candidate* should replace *current* as active.

    When ``policy.min_improvement`` is positive, the candidate's
    ``macro_f1`` must exceed the current active model's ``macro_f1``
    by at least that amount.  If ``current`` is ``None`` or has no
    recorded ``macro_f1``, the switch is always allowed.

    Args:
        current: The current active pointer (may be ``None``).
        candidate: The best-ranked bundle from selection.
        policy: Selection policy with hysteresis threshold.

    Returns:
        ``True`` if the active pointer should be updated.
    """
    if current is None or policy.min_improvement <= 0.0:
        return True

    if candidate.metrics is None:
        return False

    current_f1: float | None = None
    if current.reason and "macro_f1" in current.reason:
        try:
            current_f1 = float(current.reason["macro_f1"])  # type: ignore[arg-type]
        except (TypeError, ValueError):
            pass

    if current_f1 is None:
        return True

    return candidate.metrics.macro_f1 >= current_f1 + policy.min_improvement
