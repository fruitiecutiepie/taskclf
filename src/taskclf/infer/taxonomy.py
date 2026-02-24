"""User-specific taxonomy mapping: core labels -> user-defined buckets.

This module implements the personalization mapping layer described in
``docs/guide/model_io.md`` Section 5.  It converts model predictions
(core label + probability vector) into user-facing bucket labels with
aggregated probabilities, without altering the underlying core predictions.

Typical flow::

    config = load_taxonomy(Path("configs/user_taxonomy.yaml"))
    resolver = TaxonomyResolver(config)
    result = resolver.resolve(core_label_id, core_probs)
    # result.mapped_label, result.mapped_probs
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Final, Literal, Sequence

import numpy as np
import yaml
from pydantic import BaseModel, Field, model_validator

from taskclf.core.defaults import MIXED_UNKNOWN
from taskclf.core.types import LABEL_SET_V1, CoreLabel

logger = logging.getLogger(__name__)

_HEX_COLOR_RE: Final[re.Pattern[str]] = re.compile(r"^#[0-9A-Fa-f]{6}$")
_CORE_LABEL_NAMES: Final[list[str]] = sorted(CoreLabel)
_CORE_LABEL_INDEX: Final[dict[str, int]] = {
    name: idx for idx, name in enumerate(_CORE_LABEL_NAMES)
}
FALLBACK_BUCKET_NAME: Final[str] = "Other"


# ---------------------------------------------------------------------------
# Config models
# ---------------------------------------------------------------------------


class TaxonomyBucket(BaseModel, frozen=True):
    """A user-facing task category that aggregates one or more core labels."""

    name: str = Field(min_length=1, description="Unique display name for this bucket.")
    description: str = Field(default="", description="Human-readable description.")
    core_labels: list[str] = Field(
        min_length=1, description="Core labels mapped to this bucket."
    )
    color: str = Field(default="#808080", description="Hex color for display.")

    @model_validator(mode="after")
    def _validate(self) -> TaxonomyBucket:
        for label in self.core_labels:
            if label not in LABEL_SET_V1:
                raise ValueError(
                    f"Unknown core label {label!r} in bucket {self.name!r}; "
                    f"must be one of {_CORE_LABEL_NAMES}"
                )
        if not _HEX_COLOR_RE.match(self.color):
            raise ValueError(
                f"Invalid hex color {self.color!r} in bucket {self.name!r}; "
                f"expected format #RRGGBB"
            )
        return self


class TaxonomyDisplay(BaseModel, frozen=True):
    """User display preferences (not used by resolver logic)."""

    show_core_labels: bool = False
    default_view: Literal["mapped", "core"] = "mapped"
    color_theme: str = "default"


class TaxonomyReject(BaseModel, frozen=True):
    """How rejected predictions are surfaced."""

    mixed_label_name: str = MIXED_UNKNOWN
    include_rejected_in_reports: bool = False


class TaxonomyAdvanced(BaseModel, frozen=True):
    """Advanced mapping tuning knobs."""

    probability_aggregation: Literal["sum", "max"] = "sum"
    min_confidence_for_mapping: float = Field(
        default=0.55, ge=0.0, le=1.0
    )
    reweight_core_labels: dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_reweights(self) -> TaxonomyAdvanced:
        for label, weight in self.reweight_core_labels.items():
            if label not in LABEL_SET_V1:
                raise ValueError(
                    f"Unknown core label {label!r} in reweight_core_labels; "
                    f"must be one of {_CORE_LABEL_NAMES}"
                )
            if weight <= 0:
                raise ValueError(
                    f"Reweight for {label!r} must be > 0, got {weight}"
                )
        return self


class TaxonomyConfig(BaseModel, frozen=True):
    """Full user-specific taxonomy mapping configuration.

    Loaded from a YAML file matching the format in
    ``configs/user_taxonomy_example.yaml``.
    """

    version: str = "1.0"
    label_schema_version: str = "labels_v1"
    user_id: str | None = None
    display: TaxonomyDisplay = Field(default_factory=TaxonomyDisplay)
    reject: TaxonomyReject = Field(default_factory=TaxonomyReject)
    buckets: list[TaxonomyBucket] = Field(min_length=1)
    advanced: TaxonomyAdvanced = Field(default_factory=TaxonomyAdvanced)

    @model_validator(mode="after")
    def _validate_config(self) -> TaxonomyConfig:
        names = [b.name for b in self.buckets]
        if len(names) != len(set(names)):
            seen: set[str] = set()
            dupes = [n for n in names if n in seen or seen.add(n)]  # type: ignore[func-returns-value]
            raise ValueError(f"Duplicate bucket names: {dupes}")
        return self


# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------


class TaxonomyResult(BaseModel, frozen=True):
    """Output of the taxonomy mapping resolver for a single window."""

    mapped_label: str = Field(description="User-facing bucket label.")
    mapped_probs: dict[str, float] = Field(
        description="Bucket name -> aggregated probability (sums to 1.0)."
    )


# ---------------------------------------------------------------------------
# YAML I/O
# ---------------------------------------------------------------------------


def load_taxonomy(path: Path) -> TaxonomyConfig:
    """Load and validate a taxonomy config from a YAML file.

    Args:
        path: Path to a YAML file matching the taxonomy config schema.

    Returns:
        Validated ``TaxonomyConfig``.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError / ValidationError: If the YAML is malformed or invalid.
    """
    raw = yaml.safe_load(path.read_text())
    if isinstance(raw, dict) and "version" in raw:
        raw["version"] = str(raw["version"])
    return TaxonomyConfig.model_validate(raw)


def save_taxonomy(config: TaxonomyConfig, path: Path) -> Path:
    """Serialize a taxonomy config to YAML.

    Args:
        config: Validated taxonomy config to write.
        path: Destination file path.

    Returns:
        The *path* that was written.
    """
    data = config.model_dump(mode="json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.dump(data, default_flow_style=False, sort_keys=False))
    return path


def default_taxonomy() -> TaxonomyConfig:
    """Create an identity taxonomy: one bucket per core label.

    Useful as a starting point for user customisation.
    """
    buckets = [
        TaxonomyBucket(
            name=label,
            description=f"Core label: {label}",
            core_labels=[label],
        )
        for label in CoreLabel
    ]
    return TaxonomyConfig(buckets=buckets)


# ---------------------------------------------------------------------------
# Resolver
# ---------------------------------------------------------------------------


class TaxonomyResolver:
    """Stateless mapper from core predictions to user-defined buckets.

    Precomputes index lookups at construction time so that per-row
    resolution is fast.

    Args:
        config: Validated taxonomy config.
    """

    def __init__(self, config: TaxonomyConfig) -> None:
        self.config = config

        self._bucket_names: list[str] = [b.name for b in config.buckets]
        self._bucket_core_indices: list[list[int]] = []
        covered: set[str] = set()
        for bucket in config.buckets:
            indices = [_CORE_LABEL_INDEX[lbl] for lbl in bucket.core_labels]
            self._bucket_core_indices.append(indices)
            covered.update(bucket.core_labels)

        uncovered = LABEL_SET_V1 - covered
        self._has_fallback = bool(uncovered)
        if self._has_fallback:
            logger.info(
                "Core labels %s not in any bucket; assigning to '%s' fallback",
                sorted(uncovered),
                FALLBACK_BUCKET_NAME,
            )
            fallback_indices = [_CORE_LABEL_INDEX[lbl] for lbl in sorted(uncovered)]
            self._bucket_names.append(FALLBACK_BUCKET_NAME)
            self._bucket_core_indices.append(fallback_indices)

        self._n_buckets = len(self._bucket_names)
        self._agg = config.advanced.probability_aggregation

        self._reweights: np.ndarray | None = None
        if config.advanced.reweight_core_labels:
            w = np.ones(len(_CORE_LABEL_NAMES), dtype=np.float64)
            for label, weight in config.advanced.reweight_core_labels.items():
                w[_CORE_LABEL_INDEX[label]] = weight
            self._reweights = w

    def resolve(
        self,
        core_label_id: int,  # noqa: ARG002 – kept for API symmetry
        core_probs: np.ndarray,
        *,
        is_rejected: bool = False,
    ) -> TaxonomyResult:
        """Map a single window's core prediction to a user bucket.

        Args:
            core_label_id: Index of the predicted core label (unused
                directly -- probabilities drive the mapping).
            core_probs: Probability vector of shape ``(8,)`` from the
                model.  **Not modified** by this method.
            is_rejected: Whether the prediction was below the reject
                threshold.

        Returns:
            A ``TaxonomyResult`` with ``mapped_label`` and
            ``mapped_probs``.
        """
        if is_rejected:
            return TaxonomyResult(
                mapped_label=self.config.reject.mixed_label_name,
                mapped_probs={},
            )

        probs = core_probs.astype(np.float64, copy=True)

        if self._reweights is not None:
            probs = probs * self._reweights
            total = probs.sum()
            if total > 0:
                probs /= total

        bucket_probs = np.zeros(self._n_buckets, dtype=np.float64)
        for i, indices in enumerate(self._bucket_core_indices):
            if self._agg == "sum":
                bucket_probs[i] = probs[indices].sum()
            else:
                bucket_probs[i] = probs[indices].max()

        bp_total = bucket_probs.sum()
        if bp_total > 0:
            bucket_probs /= bp_total

        best_idx = int(bucket_probs.argmax())
        mapped_label = self._bucket_names[best_idx]
        mapped_probs = {
            name: round(float(p), 6)
            for name, p in zip(self._bucket_names, bucket_probs)
        }

        return TaxonomyResult(mapped_label=mapped_label, mapped_probs=mapped_probs)

    def resolve_batch(
        self,
        core_label_ids: np.ndarray,
        core_probs: np.ndarray,
        *,
        is_rejected: np.ndarray | None = None,
    ) -> list[TaxonomyResult]:
        """Map a batch of core predictions to user buckets.

        Args:
            core_label_ids: Shape ``(N,)`` array of predicted core label
                indices.
            core_probs: Shape ``(N, 8)`` probability matrix.
            is_rejected: Optional boolean array of shape ``(N,)``.

        Returns:
            List of ``TaxonomyResult``, one per row.
        """
        n = len(core_label_ids)
        if is_rejected is None:
            is_rejected = np.zeros(n, dtype=bool)
        return [
            self.resolve(
                int(core_label_ids[i]),
                core_probs[i],
                is_rejected=bool(is_rejected[i]),
            )
            for i in range(n)
        ]

    @property
    def bucket_names(self) -> list[str]:
        """Ordered list of bucket names (including fallback if present)."""
        return list(self._bucket_names)
