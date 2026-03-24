"""Versioned inference-policy artifact: model + calibration + reject threshold.

The :class:`InferencePolicy` is the canonical deployment descriptor.  It
binds a specific model bundle to an optional calibrator store and a
reject threshold that was tuned on the (potentially calibrated) score
distribution.

Persistence uses the same atomic-write pattern as
:func:`~taskclf.model_registry.write_active_atomic`: write to a
temporary file, then :func:`os.replace` so readers never see a partial
write.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from taskclf.core.defaults import (
    DEFAULT_GIT_TIMEOUT_SECONDS,
    DEFAULT_INFERENCE_POLICY_FILE,
    DEFAULT_REJECT_THRESHOLD,
)

logger = logging.getLogger(__name__)

_POLICY_TMP = ".inference_policy.json.tmp"


class InferencePolicy(BaseModel, frozen=True):
    """Versioned deployment descriptor binding model + calibration + threshold.

    Stored as ``models/inference_policy.json``.  Inference resolution
    reads this file to determine which model bundle, calibrator store,
    and reject threshold to use.

    All paths are **relative to ``models_dir.parent``** (i.e. relative
    to ``TASKCLF_HOME``), matching the convention used by
    :class:`~taskclf.model_registry.ActivePointer`.
    """

    policy_version: Literal["v1"] = "v1"

    # ── Model binding ──
    model_dir: str
    model_schema_hash: str
    model_label_set: list[str]

    # ── Calibration binding (None = identity / no calibration) ──
    calibrator_store_dir: str | None = None
    calibration_method: str | None = None

    # ── Reject threshold (tuned on calibrated scores when calibrator is present) ──
    reject_threshold: float

    # ── Provenance ──
    created_at: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat(),
    )
    source: str = "manual"
    git_commit: str = ""


def _current_git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=DEFAULT_GIT_TIMEOUT_SECONDS,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        logger.debug("Could not determine git commit", exc_info=True)
        return "unknown"


def build_inference_policy(
    *,
    model_dir: str,
    model_schema_hash: str,
    model_label_set: list[str],
    reject_threshold: float = DEFAULT_REJECT_THRESHOLD,
    calibrator_store_dir: str | None = None,
    calibration_method: str | None = None,
    source: str = "manual",
) -> InferencePolicy:
    """Convenience builder that fills in provenance fields automatically.

    Args:
        model_dir: Path to model bundle, relative to ``models_dir.parent``.
        model_schema_hash: Schema hash from the model's metadata.
        model_label_set: Label set from the model's metadata.
        reject_threshold: Reject threshold for this model+calibration pair.
        calibrator_store_dir: Path to calibrator store directory,
            relative to ``models_dir.parent``.  ``None`` for identity.
        calibration_method: ``"temperature"`` or ``"isotonic"``; ``None``
            when no calibrator store is used.
        source: How this policy was created (``"manual"``,
            ``"tune-reject"``, ``"retrain"``, ``"calibrate"``).

    Returns:
        A populated :class:`InferencePolicy`.
    """
    return InferencePolicy(
        model_dir=model_dir,
        model_schema_hash=model_schema_hash,
        model_label_set=sorted(model_label_set),
        reject_threshold=reject_threshold,
        calibrator_store_dir=calibrator_store_dir,
        calibration_method=calibration_method,
        source=source,
        git_commit=_current_git_commit(),
    )


def save_inference_policy(
    policy: InferencePolicy,
    models_dir: Path,
) -> Path:
    """Atomically persist *policy* as ``models_dir/inference_policy.json``.

    Uses temp-file + :func:`os.replace` so readers never see a partial
    write.

    Args:
        policy: The policy to persist.
        models_dir: The ``models/`` directory.

    Returns:
        Path to the written policy file.
    """
    models_dir.mkdir(parents=True, exist_ok=True)

    tmp_path = models_dir / _POLICY_TMP
    final_path = models_dir / DEFAULT_INFERENCE_POLICY_FILE

    tmp_path.write_text(policy.model_dump_json(indent=2) + "\n")
    os.replace(tmp_path, final_path)

    logger.info(
        "Wrote inference policy to %s (model=%s, threshold=%.4f, source=%s)",
        final_path,
        policy.model_dir,
        policy.reject_threshold,
        policy.source,
    )
    return final_path


def load_inference_policy(models_dir: Path) -> InferencePolicy | None:
    """Load the inference policy from ``models_dir/inference_policy.json``.

    Returns ``None`` (without raising) when the file is missing, contains
    invalid JSON, or fails validation.

    Args:
        models_dir: The ``models/`` directory.

    Returns:
        A validated :class:`InferencePolicy`, or ``None``.
    """
    policy_path = models_dir / DEFAULT_INFERENCE_POLICY_FILE
    if not policy_path.is_file():
        return None

    try:
        raw = json.loads(policy_path.read_text())
        return InferencePolicy.model_validate(raw)
    except (json.JSONDecodeError, ValueError, OSError) as exc:
        logger.warning(
            "Could not load inference policy from %s: %s",
            policy_path,
            exc,
        )
        return None


def remove_inference_policy(models_dir: Path) -> bool:
    """Delete ``models_dir/inference_policy.json`` if it exists.

    Args:
        models_dir: The ``models/`` directory.

    Returns:
        ``True`` if the file was removed, ``False`` if it did not exist.
    """
    policy_path = models_dir / DEFAULT_INFERENCE_POLICY_FILE
    try:
        policy_path.unlink()
        logger.info("Removed inference policy at %s", policy_path)
        return True
    except FileNotFoundError:
        return False


class PolicyValidationError(Exception):
    """Raised when an inference policy fails validation against disk artifacts."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.details = details or {}


def validate_policy(
    policy: InferencePolicy,
    models_dir: Path,
) -> None:
    """Validate that *policy* references artifacts that exist and are compatible.

    Checks:

    1. ``model_dir`` resolves to an existing directory with ``metadata.json``.
    2. ``model_schema_hash`` matches the bundle's recorded schema hash.
    3. ``model_label_set`` matches the bundle's recorded label set.
    4. If ``calibrator_store_dir`` is set, it resolves to an existing
       directory with ``store.json``.
    5. If the calibrator store has model-binding metadata, it matches the
       policy's model binding.

    Args:
        policy: The policy to validate.
        models_dir: The ``models/`` directory (paths are resolved
            relative to ``models_dir.parent``).

    Raises:
        PolicyValidationError: When any check fails.
    """
    base = models_dir.parent
    bundle_path = base / policy.model_dir
    if not bundle_path.is_dir():
        raise PolicyValidationError(
            f"Model directory does not exist: {bundle_path}",
            {"model_dir": policy.model_dir},
        )

    meta_path = bundle_path / "metadata.json"
    if not meta_path.is_file():
        raise PolicyValidationError(
            f"Model metadata not found: {meta_path}",
            {"model_dir": policy.model_dir},
        )

    try:
        meta = json.loads(meta_path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        raise PolicyValidationError(
            f"Could not read model metadata: {exc}",
            {"model_dir": policy.model_dir},
        ) from exc

    bundle_hash = meta.get("schema_hash", "")
    if bundle_hash != policy.model_schema_hash:
        raise PolicyValidationError(
            f"Schema hash mismatch: policy has {policy.model_schema_hash!r}, "
            f"bundle has {bundle_hash!r}",
            {
                "policy_hash": policy.model_schema_hash,
                "bundle_hash": bundle_hash,
            },
        )

    bundle_labels = sorted(meta.get("label_set", []))
    if bundle_labels != sorted(policy.model_label_set):
        raise PolicyValidationError(
            f"Label set mismatch: policy has {sorted(policy.model_label_set)!r}, "
            f"bundle has {bundle_labels!r}",
        )

    if policy.calibrator_store_dir is not None:
        store_path = base / policy.calibrator_store_dir
        if not store_path.is_dir():
            raise PolicyValidationError(
                f"Calibrator store directory does not exist: {store_path}",
                {"calibrator_store_dir": policy.calibrator_store_dir},
            )
        store_meta_path = store_path / "store.json"
        if not store_meta_path.is_file():
            raise PolicyValidationError(
                f"Calibrator store metadata not found: {store_meta_path}",
                {"calibrator_store_dir": policy.calibrator_store_dir},
            )

        try:
            store_meta = json.loads(store_meta_path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            raise PolicyValidationError(
                f"Could not read calibrator store metadata: {exc}",
                {"calibrator_store_dir": policy.calibrator_store_dir},
            ) from exc

        store_model_hash = store_meta.get("model_schema_hash")
        if (
            store_model_hash is not None
            and store_model_hash != policy.model_schema_hash
        ):
            raise PolicyValidationError(
                f"Calibrator store was fitted against a different schema: "
                f"store has {store_model_hash!r}, "
                f"policy expects {policy.model_schema_hash!r}",
                {
                    "store_schema_hash": store_model_hash,
                    "policy_schema_hash": policy.model_schema_hash,
                },
            )
