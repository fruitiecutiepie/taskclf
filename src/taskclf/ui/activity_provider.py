"""Provider-neutral activity source helpers for UI-facing runtime paths."""

from __future__ import annotations

import datetime as dt
import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Literal, Protocol

from taskclf.adapters.activitywatch import client as aw_client
from taskclf.adapters.activitywatch.types import AWEvent
from taskclf.core.defaults import DEFAULT_AW_TIMEOUT_SECONDS
from taskclf.ui.copy import (
    activity_source_setup_help_url,
    activity_source_setup_message,
    activity_source_setup_steps,
    activity_source_setup_title,
)

ActivityProviderState = Literal["checking", "ready", "setup_required"]


class ActivityProviderUnavailableError(RuntimeError):
    """Raised when an activity source cannot provide data right now."""

    def __init__(
        self,
        message: str,
        *,
        retryable: bool,
        source_lost: bool = False,
    ) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.source_lost = source_lost


@dataclass(frozen=True, slots=True)
class ActivityProviderStatus:
    """UI-facing status for the configured activity source."""

    provider_id: str
    provider_name: str
    state: ActivityProviderState
    summary_available: bool
    endpoint: str
    source_id: str | None
    last_sample_count: int
    last_sample_breakdown: dict[str, int]
    setup_title: str
    setup_message: str
    setup_steps: list[str]
    help_url: str

    def to_payload(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ActivitySummaryAppEntry:
    """One activity-source app summary row."""

    app: str
    events: int

    def to_payload(self) -> dict[str, object]:
        return asdict(self)


class ActivityWatcherProvider(Protocol):
    """Contract for UI-facing activity source providers."""

    provider_id: str
    provider_name: str
    endpoint: str

    def initial_status(self) -> ActivityProviderStatus: ...

    def checking_status(
        self,
        *,
        source_id: str | None = None,
        last_sample_count: int = 0,
        last_sample_breakdown: dict[str, int] | None = None,
    ) -> ActivityProviderStatus: ...

    def ready_status(
        self,
        *,
        source_id: str | None,
        last_sample_count: int = 0,
        last_sample_breakdown: dict[str, int] | None = None,
    ) -> ActivityProviderStatus: ...

    def setup_required_status(
        self,
        *,
        source_id: str | None = None,
        last_sample_count: int = 0,
        last_sample_breakdown: dict[str, int] | None = None,
    ) -> ActivityProviderStatus: ...

    def probe_status(
        self,
        *,
        timeout_seconds: float | int | None = None,
    ) -> ActivityProviderStatus: ...

    def discover_source_id(
        self,
        *,
        timeout_seconds: float | int | None = None,
    ) -> str: ...

    def fetch_events(
        self,
        source_id: str,
        start: dt.datetime,
        end: dt.datetime,
        *,
        timeout_seconds: float | int | None = None,
    ) -> list[AWEvent]: ...

    def recent_app_summary(
        self,
        start: dt.datetime,
        end: dt.datetime,
        *,
        source_id: str | None = None,
        timeout_seconds: float | int | None = None,
    ) -> tuple[ActivityProviderStatus, list[ActivitySummaryAppEntry]]: ...


def _timeout_seconds(value: float | int | None, default: int) -> int:
    if value is None:
        return default
    if value <= 0:
        return 1
    return max(1, math.ceil(float(value)))


def find_window_bucket_id(endpoint: str, *, timeout: int) -> str:
    """Late-bound wrapper kept patchable for tests."""
    return aw_client.find_window_bucket_id(endpoint, timeout=timeout)


def fetch_aw_events(
    endpoint: str,
    source_id: str,
    start: dt.datetime,
    end: dt.datetime,
    *,
    title_salt: str,
    timeout: int,
) -> list[AWEvent]:
    """Late-bound wrapper kept patchable for tests."""
    return aw_client.fetch_aw_events(
        endpoint,
        source_id,
        start,
        end,
        title_salt=title_salt,
        timeout=timeout,
    )


@dataclass(slots=True)
class ActivityWatchProvider:
    """ActivityWatch-backed implementation of the UI activity-source contract."""

    endpoint: str
    title_salt: str
    timeout_seconds: int = DEFAULT_AW_TIMEOUT_SECONDS
    provider_id: str = "activitywatch"
    provider_name: str = "ActivityWatch"

    def _status(
        self,
        state: ActivityProviderState,
        *,
        source_id: str | None = None,
        last_sample_count: int = 0,
        last_sample_breakdown: dict[str, int] | None = None,
    ) -> ActivityProviderStatus:
        return ActivityProviderStatus(
            provider_id=self.provider_id,
            provider_name=self.provider_name,
            state=state,
            summary_available=(state == "ready"),
            endpoint=self.endpoint,
            source_id=source_id,
            last_sample_count=last_sample_count,
            last_sample_breakdown=dict(last_sample_breakdown or {}),
            setup_title=activity_source_setup_title(),
            setup_message=activity_source_setup_message(),
            setup_steps=activity_source_setup_steps(self.endpoint),
            help_url=activity_source_setup_help_url(),
        )

    def initial_status(self) -> ActivityProviderStatus:
        return self.checking_status()

    def checking_status(
        self,
        *,
        source_id: str | None = None,
        last_sample_count: int = 0,
        last_sample_breakdown: dict[str, int] | None = None,
    ) -> ActivityProviderStatus:
        return self._status(
            "checking",
            source_id=source_id,
            last_sample_count=last_sample_count,
            last_sample_breakdown=last_sample_breakdown,
        )

    def ready_status(
        self,
        *,
        source_id: str | None,
        last_sample_count: int = 0,
        last_sample_breakdown: dict[str, int] | None = None,
    ) -> ActivityProviderStatus:
        return self._status(
            "ready",
            source_id=source_id,
            last_sample_count=last_sample_count,
            last_sample_breakdown=last_sample_breakdown,
        )

    def setup_required_status(
        self,
        *,
        source_id: str | None = None,
        last_sample_count: int = 0,
        last_sample_breakdown: dict[str, int] | None = None,
    ) -> ActivityProviderStatus:
        return self._status(
            "setup_required",
            source_id=source_id,
            last_sample_count=last_sample_count,
            last_sample_breakdown=last_sample_breakdown,
        )

    def probe_status(
        self,
        *,
        timeout_seconds: float | int | None = None,
    ) -> ActivityProviderStatus:
        try:
            source_id = self.discover_source_id(timeout_seconds=timeout_seconds)
        except ActivityProviderUnavailableError:
            return self.setup_required_status()
        return self.ready_status(source_id=source_id)

    def discover_source_id(
        self,
        *,
        timeout_seconds: float | int | None = None,
    ) -> str:
        timeout = _timeout_seconds(timeout_seconds, self.timeout_seconds)
        try:
            return find_window_bucket_id(self.endpoint, timeout=timeout)
        except aw_client.AWConnectionError as exc:
            raise ActivityProviderUnavailableError(str(exc), retryable=True) from exc
        except aw_client.AWTimeoutError as exc:
            raise ActivityProviderUnavailableError(str(exc), retryable=False) from exc
        except aw_client.AWNotFoundError as exc:
            raise ActivityProviderUnavailableError(str(exc), retryable=True) from exc
        except ValueError as exc:
            raise ActivityProviderUnavailableError(str(exc), retryable=False) from exc

    def fetch_events(
        self,
        source_id: str,
        start: dt.datetime,
        end: dt.datetime,
        *,
        timeout_seconds: float | int | None = None,
    ) -> list[AWEvent]:
        timeout = _timeout_seconds(timeout_seconds, self.timeout_seconds)
        try:
            return fetch_aw_events(
                self.endpoint,
                source_id,
                start,
                end,
                title_salt=self.title_salt,
                timeout=timeout,
            )
        except aw_client.AWConnectionError as exc:
            raise ActivityProviderUnavailableError(str(exc), retryable=True) from exc
        except aw_client.AWTimeoutError as exc:
            raise ActivityProviderUnavailableError(str(exc), retryable=False) from exc
        except aw_client.AWNotFoundError as exc:
            raise ActivityProviderUnavailableError(
                str(exc),
                retryable=True,
                source_lost=True,
            ) from exc
        except ValueError as exc:
            raise ActivityProviderUnavailableError(str(exc), retryable=False) from exc

    def recent_app_summary(
        self,
        start: dt.datetime,
        end: dt.datetime,
        *,
        source_id: str | None = None,
        timeout_seconds: float | int | None = None,
    ) -> tuple[ActivityProviderStatus, list[ActivitySummaryAppEntry]]:
        resolved_source_id = source_id or self.discover_source_id(
            timeout_seconds=timeout_seconds
        )
        events = self.fetch_events(
            resolved_source_id,
            start,
            end,
            timeout_seconds=timeout_seconds,
        )
        counts = Counter(event.app_id for event in events)
        breakdown = dict(counts.most_common(5))
        status = self.ready_status(
            source_id=resolved_source_id,
            last_sample_count=len(events),
            last_sample_breakdown=breakdown,
        )
        return (
            status,
            [
                ActivitySummaryAppEntry(app=app_id, events=count)
                for app_id, count in counts.most_common(5)
            ],
        )


def summarize_events_by_app(events: Sequence[AWEvent]) -> dict[str, int]:
    """Return a top-app breakdown for monitor diagnostics."""
    return dict(Counter(event.app_id for event in events).most_common(5))
