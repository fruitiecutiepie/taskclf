"""Shared non-GUI runtime pieces for the UI and tray flows."""

from __future__ import annotations

import datetime as dt
import logging
import threading
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from taskclf.core.defaults import (
    DEFAULT_AW_HOST,
    DEFAULT_AW_TIMEOUT_SECONDS,
    DEFAULT_IDLE_TRANSITION_MINUTES,
    DEFAULT_POLL_SECONDS,
    DEFAULT_TITLE_SALT,
    DEFAULT_TRANSITION_MINUTES,
)
from taskclf.ui.events import EventBus

# Keep the historical logger namespace stable even though the runtime helpers
# no longer live in the tray module.
logger = logging.getLogger("taskclf.ui.tray")

_MAX_BACKOFF_SECONDS = 300
_WARN_AFTER_FAILURES = 3

_LOCKSCREEN_APP_IDS: frozenset[str] = frozenset(
    {
        "com.apple.loginwindow",
        "com.microsoft.LockApp",
        "com.microsoft.LogonUI",
        "org.gnome.ScreenSaver",
        "org.gnome.Shell",
        "org.freedesktop.portal.Desktop",
        "org.i3wm.i3lock",
        "org.swaywm.swaylock",
        "org.jwz.xscreensaver",
        "org.freedesktop.light-locker",
        "org.suckless.slock",
    }
)


@dataclass(kw_only=True, eq=False)
class ActivityMonitor:
    """Poll ActivityWatch and emit transitions when the dominant app changes."""

    aw_host: str = DEFAULT_AW_HOST
    title_salt: str = DEFAULT_TITLE_SALT
    poll_seconds: int = DEFAULT_POLL_SECONDS
    aw_timeout_seconds: int = DEFAULT_AW_TIMEOUT_SECONDS
    transition_minutes: int = DEFAULT_TRANSITION_MINUTES
    idle_transition_minutes: int = DEFAULT_IDLE_TRANSITION_MINUTES
    on_transition: Callable[[str, str, dt.datetime, dt.datetime], Any] | None = None
    on_poll: Callable[[str], Any] | None = None
    on_initial_app: Callable[[str, dt.datetime], Any] | None = None
    event_bus: EventBus | None = None
    _aw_host: str = field(init=False)
    _title_salt: str = field(init=False)
    _poll_seconds: int = field(init=False)
    _aw_timeout_seconds: int = field(init=False)
    _transition_threshold: int = field(init=False)
    _idle_transition_threshold: int = field(init=False)
    _on_transition: Callable[[str, str, dt.datetime, dt.datetime], Any] | None = field(
        init=False, default=None
    )
    _on_poll: Callable[[str], Any] | None = field(init=False, default=None)
    _on_initial_app: Callable[[str, dt.datetime], Any] | None = field(
        init=False, default=None
    )
    _event_bus: EventBus | None = field(init=False, default=None)
    _current_app: str | None = field(init=False, default=None)
    _current_app_since: dt.datetime | None = field(init=False, default=None)
    _candidate_app: str | None = field(init=False, default=None)
    _candidate_duration: int = field(init=False, default=0)
    _candidate_first_seen: dt.datetime | None = field(init=False, default=None)
    _last_check_time: dt.datetime | None = field(init=False, default=None)
    _bucket_id: str | None = field(init=False, default=None)
    _aw_warned: bool = field(init=False, default=False)
    _stop: threading.Event = field(init=False, default_factory=threading.Event)
    _paused: threading.Event = field(init=False, default_factory=threading.Event)
    _poll_count: int = field(init=False, default=0)
    _last_event_count: int = field(init=False, default=0)
    _last_app_counts: dict[str, int] = field(init=False, default_factory=dict)
    _last_poll_ts: dt.datetime | None = field(init=False, default=None)
    _started_at: dt.datetime | None = field(init=False, default=None)
    _consecutive_failures: int = field(init=False, default=0)
    _backoff_seconds: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        self._aw_host = self.aw_host
        self._title_salt = self.title_salt
        self._poll_seconds = self.poll_seconds
        self._aw_timeout_seconds = self.aw_timeout_seconds
        self._transition_threshold = self.transition_minutes * 60
        self._idle_transition_threshold = self.idle_transition_minutes * 60
        self._on_transition = self.on_transition
        self._on_poll = self.on_poll
        self._on_initial_app = self.on_initial_app
        self._event_bus = self.event_bus

    def _discover_bucket(self) -> str:
        from taskclf.adapters.activitywatch.client import find_window_bucket_id

        return find_window_bucket_id(self._aw_host, timeout=self._aw_timeout_seconds)

    def _after_fetch_failure(self, exc: Exception) -> None:
        """Update retry state after a failed ActivityWatch request."""
        from taskclf.adapters.activitywatch.client import AWConnectionError

        self._consecutive_failures += 1
        if isinstance(exc, AWConnectionError):
            self._backoff_seconds = min(
                self._poll_seconds * (2**self._consecutive_failures),
                _MAX_BACKOFF_SECONDS,
            )
        else:
            self._backoff_seconds = 0

        if self._consecutive_failures == _WARN_AFTER_FAILURES:
            logger.warning(
                "ActivityWatch unreachable (%d consecutive failures): %s",
                self._consecutive_failures,
                exc,
            )
            self._publish_status(self._current_app or "unknown", state="aw_unreachable")
        elif self._consecutive_failures > _WARN_AFTER_FAILURES:
            logger.debug(
                "ActivityWatch still unreachable (%d failures)",
                self._consecutive_failures,
            )

    def _after_fetch_success(self) -> None:
        """Reset retry state after a successful ActivityWatch request."""
        if self._consecutive_failures >= _WARN_AFTER_FAILURES:
            logger.info(
                "ActivityWatch connection restored after %d failures",
                self._consecutive_failures,
            )
        self._consecutive_failures = 0
        self._backoff_seconds = 0

    def _poll_dominant_app(self) -> str | None:
        """Fetch recent ActivityWatch events and return the most common app id."""
        from taskclf.adapters.activitywatch.client import (
            AWConnectionError,
            AWNotFoundError,
            AWTimeoutError,
            fetch_aw_events,
        )

        if self._bucket_id is None:
            try:
                self._bucket_id = self._discover_bucket()
                if self._aw_warned:
                    print(f"Connected to ActivityWatch at {self._aw_host}")
                    self._aw_warned = False
                self._after_fetch_success()
            except (AWConnectionError, AWTimeoutError) as exc:
                self._after_fetch_failure(exc)
                if not self._aw_warned:
                    print(
                        f"Waiting for ActivityWatch at {self._aw_host} "
                        f"(retrying every {self._poll_seconds}s)..."
                    )
                    self._aw_warned = True
                self._last_event_count = 0
                self._last_app_counts = {}
                return None
            except Exception:
                if not self._aw_warned:
                    print(
                        f"Waiting for ActivityWatch at {self._aw_host} "
                        f"(retrying every {self._poll_seconds}s)..."
                    )
                    self._aw_warned = True
                self._last_event_count = 0
                self._last_app_counts = {}
                return None

        now = dt.datetime.now(dt.timezone.utc)
        start = now - dt.timedelta(seconds=self._poll_seconds)
        try:
            events = fetch_aw_events(
                self._aw_host,
                self._bucket_id,
                start,
                now,
                title_salt=self._title_salt,
                timeout=self._aw_timeout_seconds,
            )
        except AWNotFoundError as exc:
            logger.warning("ActivityWatch bucket not found, will rediscover: %s", exc)
            self._bucket_id = None
            self._last_event_count = 0
            self._last_app_counts = {}
            return None
        except (AWConnectionError, AWTimeoutError) as exc:
            self._after_fetch_failure(exc)
            logger.debug("Failed to fetch AW events: %s", exc)
            self._last_event_count = 0
            self._last_app_counts = {}
            return None
        except Exception:
            self._consecutive_failures += 1
            logger.debug("Failed to fetch AW events", exc_info=True)
            self._last_event_count = 0
            self._last_app_counts = {}
            return None

        self._after_fetch_success()

        if not events:
            self._last_event_count = 0
            self._last_app_counts = {}
            return None

        counts = Counter(ev.app_id for ev in events)
        self._last_event_count = len(events)
        self._last_app_counts = dict(counts.most_common(5))
        return counts.most_common(1)[0][0]

    def check_transition(
        self,
        dominant_app: str,
        *,
        _now: dt.datetime | None = None,
    ) -> None:
        """Update transition state and fire the callback when warranted."""
        now = _now or dt.datetime.now(dt.timezone.utc)
        elapsed = (
            int((now - self._last_check_time).total_seconds())
            if self._last_check_time is not None
            else self._poll_seconds
        )
        self._last_check_time = now

        if self._current_app is None:
            self._current_app = dominant_app
            self._current_app_since = now
            logger.debug("DEBUG poll: initial app=%r", dominant_app)
            if self._on_initial_app is not None:
                self._on_initial_app(dominant_app, now)
            return

        if dominant_app != self._current_app:
            leaving_lockscreen = (
                self._current_app in _LOCKSCREEN_APP_IDS
                and dominant_app not in _LOCKSCREEN_APP_IDS
            )
            if leaving_lockscreen:
                block_start = self._current_app_since or now
                block_end = now
                prev = self._current_app

                logger.debug(
                    "DEBUG poll: IMMEDIATE idle->active transition %r -> %r (block %s -> %s)",
                    prev,
                    dominant_app,
                    block_start.isoformat(),
                    block_end.isoformat(),
                )

                self._current_app = dominant_app
                self._current_app_since = block_end
                self._candidate_app = None
                self._candidate_duration = 0
                self._candidate_first_seen = None

                if self._on_transition is not None:
                    self._on_transition(prev, dominant_app, block_start, block_end)
            elif self._candidate_app == dominant_app:
                self._candidate_duration += elapsed
                is_idle_candidate = dominant_app in _LOCKSCREEN_APP_IDS
                effective_threshold = (
                    self._idle_transition_threshold
                    if is_idle_candidate
                    else self._transition_threshold
                )
                logger.debug(
                    "DEBUG poll: candidate %r held %ds / %ds threshold%s",
                    dominant_app,
                    self._candidate_duration,
                    effective_threshold,
                    " (idle)" if is_idle_candidate else "",
                )
                if self._candidate_duration >= effective_threshold:
                    block_start = self._current_app_since or now
                    block_end = self._candidate_first_seen or now
                    prev = self._current_app

                    logger.debug(
                        "DEBUG poll: TRANSITION FIRED %r -> %r (block %s -> %s)",
                        prev,
                        dominant_app,
                        block_start.isoformat(),
                        block_end.isoformat(),
                    )

                    self._current_app = dominant_app
                    self._current_app_since = block_end
                    self._candidate_app = None
                    self._candidate_duration = 0
                    self._candidate_first_seen = None

                    if self._on_transition is not None:
                        self._on_transition(prev, dominant_app, block_start, block_end)
            else:
                logger.debug(
                    "DEBUG poll: new candidate %r (was %r, current %r)",
                    dominant_app,
                    self._candidate_app,
                    self._current_app,
                )
                self._candidate_app = dominant_app
                self._candidate_duration = elapsed
                self._candidate_first_seen = now
        else:
            if self._candidate_app is not None:
                logger.debug(
                    "DEBUG poll: candidate %r reset, back to current %r",
                    self._candidate_app,
                    self._current_app,
                )
            self._candidate_app = None
            self._candidate_duration = 0
            self._candidate_first_seen = None

    def _publish_status(
        self,
        dominant_app: str,
        *,
        state: str = "collecting",
    ) -> None:
        now = dt.datetime.now(dt.timezone.utc)
        self._last_poll_ts = now
        self._poll_count += 1

        if self._event_bus is not None:
            uptime_s = (
                int((now - self._started_at).total_seconds()) if self._started_at else 0
            )
            self._event_bus.publish_threadsafe(
                {
                    "type": "status",
                    "state": state,
                    "current_app": dominant_app,
                    "current_app_since": (
                        self._current_app_since.isoformat()
                        if self._current_app_since
                        else None
                    ),
                    "candidate_app": self._candidate_app,
                    "candidate_duration_s": self._candidate_duration,
                    "transition_threshold_s": self._transition_threshold,
                    "poll_seconds": self._poll_seconds,
                    "poll_count": self._poll_count,
                    "last_poll_ts": now.isoformat(),
                    "uptime_s": uptime_s,
                    "aw_connected": self._bucket_id is not None,
                    "aw_bucket_id": self._bucket_id,
                    "aw_host": self._aw_host,
                    "last_event_count": self._last_event_count,
                    "last_app_counts": self._last_app_counts,
                }
            )

    def run(self) -> None:
        """Blocking poll loop. Call from a daemon thread."""
        if self._event_bus is not None and not self._event_bus.wait_ready(timeout=30):
            logger.warning("EventBus loop not bound after 30s, starting anyway")
        self._started_at = dt.datetime.now(dt.timezone.utc)
        while not self._stop.is_set():
            if self._paused.is_set():
                app = self._current_app or "unknown"
                self._publish_status(app, state="paused")
            else:
                dominant = self._poll_dominant_app()
                app = dominant or self._current_app or "unknown"
                if self._on_poll is not None:
                    self._on_poll(app)
                self._publish_status(app)
                if dominant is not None:
                    self.check_transition(dominant)
            self._stop.wait(timeout=self._poll_seconds + self._backoff_seconds)

    def stop(self) -> None:
        """Signal the poll loop to stop."""
        self._stop.set()

    def pause(self) -> None:
        """Pause monitoring without clearing session state."""
        self._paused.set()

    def resume(self) -> None:
        """Resume monitoring after a pause."""
        self._last_check_time = None
        self._paused.clear()

    @property
    def is_paused(self) -> bool:
        return self._paused.is_set()

    @property
    def current_app(self) -> str | None:
        return self._current_app


@dataclass(eq=False)
class _LabelSuggester:
    """Wrap the online predictor for single-interval label suggestions."""

    model_dir: Path
    _predictor: Any = field(init=False)
    _aw_host: str = field(init=False, default=DEFAULT_AW_HOST)
    _title_salt: str = field(init=False, default=DEFAULT_TITLE_SALT)
    _user_id: str = field(init=False, default="default-user")

    def __post_init__(self) -> None:
        from taskclf.core.model_io import load_model_bundle
        from taskclf.infer.online import OnlinePredictor

        logger.debug(
            "Creating _LabelSuggester via direct construction (no policy); prefer from_policy() for full inference config."
        )
        model, metadata, cat_encoders = load_model_bundle(self.model_dir)
        self._predictor = OnlinePredictor(
            model,
            metadata,
            cat_encoders=cat_encoders,
        )

    @classmethod
    def from_policy(
        cls,
        models_dir: Path,
    ) -> "_LabelSuggester":
        """Create a suggester from the active inference policy."""
        from taskclf.infer.resolve import (
            ModelResolutionError,
            resolve_inference_config,
        )

        try:
            config = resolve_inference_config(models_dir)
        except ModelResolutionError as exc:
            raise ValueError(str(exc)) from exc

        from taskclf.infer.online import OnlinePredictor

        model_path = (
            models_dir.parent / config.policy.model_dir if config.policy else None
        )
        effective_dir = model_path if model_path and model_path.is_dir() else models_dir

        obj = object.__new__(cls)
        obj.model_dir = effective_dir
        obj._aw_host = DEFAULT_AW_HOST
        obj._title_salt = DEFAULT_TITLE_SALT
        obj._user_id = "default-user"
        obj._predictor = OnlinePredictor(
            config.model,
            config.metadata,
            cat_encoders=config.cat_encoders,
            reject_threshold=config.reject_threshold,
            calibrator=config.calibrator,
            calibrator_store=config.calibrator_store,
        )
        return obj

    def suggest(
        self,
        start: dt.datetime,
        end: dt.datetime,
    ) -> tuple[str, float] | None:
        """Predict a label for the given time window."""
        from taskclf.adapters.activitywatch.client import (
            fetch_aw_events,
            fetch_aw_input_events,
            find_input_bucket_id,
            find_window_bucket_id,
        )
        from taskclf.features.build import build_features_from_aw_events
        from taskclf.infer.aggregation import aggregate_interval

        try:
            bucket_id = find_window_bucket_id(self._aw_host)
            events = fetch_aw_events(
                self._aw_host,
                bucket_id,
                start,
                end,
                title_salt=self._title_salt,
            )
            if not events:
                return None

            input_events = None
            try:
                input_bucket_id = find_input_bucket_id(self._aw_host)
                if input_bucket_id:
                    input_events = (
                        fetch_aw_input_events(
                            self._aw_host, input_bucket_id, start, end
                        )
                        or None
                    )
            except Exception:
                logger.debug(
                    "Could not fetch input events for suggestion", exc_info=True
                )

            rows = build_features_from_aw_events(
                events,
                user_id=self._user_id,
                input_events=input_events,
            )
            if not rows:
                return None

            predictions = [self._predictor.predict_bucket(row) for row in rows]
            label, confidence = aggregate_interval(predictions, strategy="majority")
            return (label, confidence)
        except Exception:
            logger.warning("Could not generate label suggestion", exc_info=True)
            return None
