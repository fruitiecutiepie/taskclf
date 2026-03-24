"""Model resolution for inference: resolve ``--model-dir`` and hot-reload.

Bridges CLI arguments to the model registry, providing:

* :func:`resolve_model_dir` — resolve an optional ``--model-dir`` to a
  concrete :class:`~pathlib.Path` using the active pointer / best-model
  selection fallback.  **Deprecated** — prefer
  :func:`resolve_inference_config`.
* :func:`resolve_inference_config` — resolve the full inference
  configuration (model + calibrator + threshold) from an
  :class:`~taskclf.core.inference_policy.InferencePolicy`.
* :class:`ActiveModelReloader` — lightweight mtime-based watcher that
  detects ``active.json`` changes and reloads the model bundle for
  long-running online inference loops.  **Deprecated** — prefer
  :class:`InferencePolicyReloader`.
* :class:`InferencePolicyReloader` — watches ``inference_policy.json``
  (falling back to ``active.json``) and reloads the full inference
  config on change.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder

from taskclf.core.defaults import (
    DEFAULT_INFERENCE_POLICY_FILE,
    DEFAULT_REJECT_THRESHOLD,
)
from taskclf.core.inference_policy import (
    InferencePolicy,
    load_inference_policy,
)
from taskclf.core.model_io import ModelMetadata, load_model_bundle
from taskclf.infer.calibration import (
    Calibrator,
    CalibratorStore,
    IdentityCalibrator,
    load_calibrator,
    load_calibrator_store,
)
from taskclf.model_registry import (
    SelectionPolicy,
    SelectionReport,
    resolve_active_model,
)

logger = logging.getLogger(__name__)

_ACTIVE_FILE = "active.json"


@dataclass(eq=False)
class ModelResolutionError(Exception):
    """Raised when no model can be resolved for inference."""

    message: str
    report: SelectionReport | None = None

    def __post_init__(self) -> None:
        super().__init__(self.message)


def resolve_model_dir(
    model_dir: str | None,
    models_dir: Path,
    policy: SelectionPolicy | None = None,
) -> Path:
    """Resolve the model directory for inference.

    .. deprecated::
        Use :func:`resolve_inference_config` instead.  This function
        only resolves the model bundle path; it does not load the
        calibrator store or reject threshold from the inference policy.

    Resolution precedence:

    1. If *model_dir* is provided, validate that it exists and return it.
    2. Otherwise, delegate to :func:`~taskclf.model_registry.resolve_active_model`
       which reads ``active.json`` or falls back to best-model selection.
    3. If no eligible model is found, raise :class:`ModelResolutionError`
       with a descriptive message including exclusion reasons.

    Args:
        model_dir: Explicit ``--model-dir`` value from CLI, or ``None``.
        models_dir: Base directory containing promoted model bundles.
        policy: Selection policy override (defaults to policy v1).

    Returns:
        Path to the resolved model bundle directory.

    Raises:
        ModelResolutionError: When no model can be resolved.
    """
    if model_dir is not None:
        path = Path(model_dir)
        if not path.is_dir():
            raise ModelResolutionError(
                f"Explicit --model-dir does not exist: {model_dir}"
            )
        return path

    if not models_dir.is_dir():
        raise ModelResolutionError(
            f"Models directory does not exist: {models_dir}. "
            "Provide --model-dir explicitly or train a model first."
        )

    bundle, report = resolve_active_model(models_dir, policy)

    if bundle is not None:
        logger.info("Resolved model: %s", bundle.path)
        return bundle.path

    lines = [
        f"No eligible model found in {models_dir}.",
        "Provide --model-dir explicitly or train a compatible model.",
    ]
    if report is not None and report.excluded:
        lines.append("Excluded bundles:")
        for rec in report.excluded:
            lines.append(f"  - {rec.model_id}: {rec.reason}")
    raise ModelResolutionError("\n".join(lines), report=report)


@dataclass(eq=False)
class ActiveModelReloader:
    """Watch ``active.json`` and reload the model bundle on change.

    .. deprecated::
        Use :class:`InferencePolicyReloader` instead.  This class only
        reloads the model bundle; it does not update the calibrator
        store or reject threshold when the policy changes.

    Designed for the online inference loop: polls the file's mtime at a
    configurable interval and, when a change is detected, loads the new
    bundle.  The caller only swaps to the new model after a successful
    load — on failure the current model is kept.

    Args:
        models_dir: Directory containing ``active.json``.
        check_interval_s: Minimum seconds between mtime checks.
    """

    models_dir: Path
    check_interval_s: float = 60.0
    _active_path: Path = field(init=False)
    _last_mtime: float | None = field(init=False)
    _last_check: float = field(init=False)

    def __post_init__(self) -> None:
        self._active_path = self.models_dir / _ACTIVE_FILE
        self._last_mtime = self._current_mtime()
        self._last_check = time.monotonic()

    def _current_mtime(self) -> float | None:
        try:
            return self._active_path.stat().st_mtime
        except OSError:
            logger.debug("Could not stat %s", self._active_path, exc_info=True)
            return None

    def check_reload(
        self,
    ) -> tuple[lgb.Booster, ModelMetadata, dict[str, Any]] | None:
        """Check whether ``active.json`` changed and reload if so.

        Returns the new ``(model, metadata, cat_encoders)`` tuple when a
        reload succeeds, or ``None`` when no reload is needed or the
        reload fails (a warning is logged on failure).
        """
        now = time.monotonic()
        if now - self._last_check < self.check_interval_s:
            return None
        self._last_check = now

        mtime = self._current_mtime()
        if mtime == self._last_mtime:
            return None

        logger.info(
            "active.json changed (mtime %s -> %s), reloading", self._last_mtime, mtime
        )

        try:
            resolved = resolve_model_dir(None, self.models_dir)
            model, metadata, cat_encoders = load_model_bundle(resolved)
        except Exception:
            logger.warning(
                "Failed to reload model after active.json change; keeping current model",
                exc_info=True,
            )
            return None

        self._last_mtime = mtime
        logger.info(
            "Reloaded model from %s (schema=%s)", resolved, metadata.schema_hash
        )
        return model, metadata, cat_encoders


# ---------------------------------------------------------------------------
# Policy-aware resolution
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResolvedInferenceConfig:
    """Fully resolved inference configuration ready for use.

    Produced by :func:`resolve_inference_config`.  Contains all
    loaded artifacts so callers do not need to perform additional I/O.
    """

    model: lgb.Booster
    metadata: ModelMetadata
    cat_encoders: dict[str, LabelEncoder]
    reject_threshold: float
    calibrator: Calibrator
    calibrator_store: CalibratorStore | None
    policy: InferencePolicy | None


def resolve_inference_config(
    models_dir: Path,
    *,
    model_dir_override: str | None = None,
    reject_threshold_override: float | None = None,
    calibrator_store_override: Path | None = None,
    calibrator_path_override: Path | None = None,
) -> ResolvedInferenceConfig:
    """Resolve the full inference configuration from policy or fallback.

    Resolution precedence:

    1. Explicit *model_dir_override* — bypasses policy; uses override
       flags for threshold and calibrator.
    2. ``models/inference_policy.json`` — loads model, calibrator store,
       and threshold from the policy.  Explicit overrides still take
       precedence for individual fields.
    3. ``models/active.json`` + code defaults — deprecated legacy
       fallback.
    4. Best-model selection + code defaults — no-config fallback.

    Args:
        models_dir: The ``models/`` directory.
        model_dir_override: Explicit ``--model-dir`` value (takes
            highest precedence).
        reject_threshold_override: Explicit threshold that overrides
            the policy value.
        calibrator_store_override: Explicit calibrator store path that
            overrides the policy value.
        calibrator_path_override: Explicit single-calibrator JSON path
            (lowest calibrator precedence).

    Returns:
        A fully resolved :class:`ResolvedInferenceConfig`.

    Raises:
        ModelResolutionError: When no model can be resolved.
    """
    policy: InferencePolicy | None = None
    model_path: Path | None = None
    reject_threshold: float = DEFAULT_REJECT_THRESHOLD
    calibrator: Calibrator = IdentityCalibrator()
    cal_store: CalibratorStore | None = None

    if model_dir_override is not None:
        model_path = Path(model_dir_override)
        if not model_path.is_dir():
            raise ModelResolutionError(
                f"Explicit --model-dir does not exist: {model_dir_override}"
            )
    else:
        policy = load_inference_policy(models_dir)
        if policy is not None:
            base = models_dir.parent
            model_path = base / policy.model_dir
            if not model_path.is_dir():
                logger.warning(
                    "Policy model_dir %s does not exist; falling back",
                    policy.model_dir,
                )
                policy = None
                model_path = None

        if model_path is None:
            if policy is not None:
                logger.warning("Policy references missing model; falling back")
                policy = None
            logger.warning(
                "No inference policy found; falling back to active.json "
                "resolution.  Create a policy with 'taskclf policy create' "
                "or 'taskclf train tune-reject --write-policy'."
            )
            model_path = resolve_model_dir(None, models_dir)

    if policy is not None:
        reject_threshold = policy.reject_threshold

    # Load calibrator store
    if calibrator_store_override is not None:
        cal_store = load_calibrator_store(calibrator_store_override)
    elif policy is not None and policy.calibrator_store_dir is not None:
        store_path = models_dir.parent / policy.calibrator_store_dir
        if store_path.is_dir():
            cal_store = load_calibrator_store(store_path)
        else:
            logger.warning(
                "Policy calibrator_store_dir %s does not exist; using identity",
                policy.calibrator_store_dir,
            )

    if calibrator_path_override is not None and cal_store is None:
        calibrator = load_calibrator(calibrator_path_override)

    if reject_threshold_override is not None:
        reject_threshold = reject_threshold_override

    model, metadata, cat_encoders = load_model_bundle(model_path)
    logger.info(
        "Resolved inference config: model=%s schema=%s threshold=%.4f "
        "calibrator_store=%s policy=%s",
        model_path.name,
        metadata.schema_hash,
        reject_threshold,
        "yes" if cal_store is not None else "no",
        "yes" if policy is not None else "legacy",
    )

    return ResolvedInferenceConfig(
        model=model,
        metadata=metadata,
        cat_encoders=cat_encoders,
        reject_threshold=reject_threshold,
        calibrator=calibrator,
        calibrator_store=cal_store,
        policy=policy,
    )


@dataclass(eq=False)
class InferencePolicyReloader:
    """Watch ``inference_policy.json`` and reload the full config on change.

    Falls back to watching ``active.json`` when no policy file exists.
    Designed for the online inference loop: polls file mtimes at a
    configurable interval and returns a :class:`ResolvedInferenceConfig`
    when a reload is needed.

    Args:
        models_dir: Directory containing policy / active pointer files.
        check_interval_s: Minimum seconds between mtime checks.
    """

    models_dir: Path
    check_interval_s: float = 60.0
    _policy_path: Path = field(init=False)
    _active_path: Path = field(init=False)
    _last_mtime: float | None = field(init=False)
    _last_check: float = field(init=False)

    def __post_init__(self) -> None:
        self._policy_path = self.models_dir / DEFAULT_INFERENCE_POLICY_FILE
        self._active_path = self.models_dir / _ACTIVE_FILE
        self._last_mtime = self._current_mtime()
        self._last_check = time.monotonic()

    def _watched_path(self) -> Path:
        return self._policy_path if self._policy_path.is_file() else self._active_path

    def _current_mtime(self) -> float | None:
        for path in (self._policy_path, self._active_path):
            try:
                return path.stat().st_mtime
            except OSError:
                continue
        return None

    def check_reload(self) -> ResolvedInferenceConfig | None:
        """Check whether the policy/active file changed and reload if so.

        Returns a :class:`ResolvedInferenceConfig` when a reload
        succeeds, or ``None`` when no reload is needed or the reload
        fails.
        """
        now = time.monotonic()
        if now - self._last_check < self.check_interval_s:
            return None
        self._last_check = now

        mtime = self._current_mtime()
        if mtime == self._last_mtime:
            return None

        watched = self._watched_path()
        logger.info(
            "%s changed (mtime %s -> %s), reloading",
            watched.name,
            self._last_mtime,
            mtime,
        )

        try:
            config = resolve_inference_config(self.models_dir)
        except Exception:
            logger.warning(
                "Failed to reload after %s change; keeping current config",
                watched.name,
                exc_info=True,
            )
            return None

        self._last_mtime = mtime
        return config
