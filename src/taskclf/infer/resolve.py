"""Model resolution for inference: resolve ``--model-dir`` and hot-reload.

Bridges CLI arguments to the model registry, providing:

* :func:`resolve_model_dir` — resolve an optional ``--model-dir`` to a
  concrete :class:`~pathlib.Path` using the active pointer / best-model
  selection fallback.
* :class:`ActiveModelReloader` — lightweight mtime-based watcher that
  detects ``active.json`` changes and reloads the model bundle for
  long-running online inference loops.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder

from taskclf.core.model_io import ModelMetadata, load_model_bundle
from taskclf.model_registry import (
    SelectionPolicy,
    SelectionReport,
    resolve_active_model,
)

logger = logging.getLogger(__name__)

_ACTIVE_FILE = "active.json"


class ModelResolutionError(Exception):
    """Raised when no model can be resolved for inference."""

    def __init__(self, message: str, report: SelectionReport | None = None) -> None:
        super().__init__(message)
        self.report = report


def resolve_model_dir(
    model_dir: str | None,
    models_dir: Path,
    policy: SelectionPolicy | None = None,
) -> Path:
    """Resolve the model directory for inference.

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


class ActiveModelReloader:
    """Watch ``active.json`` and reload the model bundle on change.

    Designed for the online inference loop: polls the file's mtime at a
    configurable interval and, when a change is detected, loads the new
    bundle.  The caller only swaps to the new model after a successful
    load — on failure the current model is kept.

    Args:
        models_dir: Directory containing ``active.json``.
        check_interval_s: Minimum seconds between mtime checks.
    """

    def __init__(
        self,
        models_dir: Path,
        check_interval_s: float = 60.0,
    ) -> None:
        self._models_dir = models_dir
        self._check_interval_s = check_interval_s
        self._active_path = models_dir / _ACTIVE_FILE
        self._last_mtime: float | None = self._current_mtime()
        self._last_check: float = time.monotonic()

    def _current_mtime(self) -> float | None:
        try:
            return self._active_path.stat().st_mtime
        except OSError:
            return None

    def check_reload(
        self,
    ) -> tuple[lgb.Booster, ModelMetadata, dict[str, LabelEncoder] | None] | None:
        """Check whether ``active.json`` changed and reload if so.

        Returns the new ``(model, metadata, cat_encoders)`` tuple when a
        reload succeeds, or ``None`` when no reload is needed or the
        reload fails (a warning is logged on failure).
        """
        now = time.monotonic()
        if now - self._last_check < self._check_interval_s:
            return None
        self._last_check = now

        mtime = self._current_mtime()
        if mtime == self._last_mtime:
            return None

        logger.info("active.json changed (mtime %s -> %s), reloading", self._last_mtime, mtime)

        try:
            resolved = resolve_model_dir(None, self._models_dir)
            model, metadata, cat_encoders = load_model_bundle(resolved)
        except Exception:
            logger.warning(
                "Failed to reload model after active.json change; keeping current model",
                exc_info=True,
            )
            return None

        self._last_mtime = mtime
        logger.info("Reloaded model from %s (schema=%s)", resolved, metadata.schema_hash)
        return model, metadata, cat_encoders
