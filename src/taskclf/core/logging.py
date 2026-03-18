"""Sanitizing log filter that redacts sensitive payloads before they reach handlers.

Prevents accidental leakage of raw window titles, keystrokes, clipboard
content, URLs, and other PII through log output.

Also provides :func:`setup_file_logging` for persisting logs to a rotating
file under ``<TASKCLF_HOME>/logs/``.
"""

from __future__ import annotations

import logging
import logging.handlers
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Final

_SENSITIVE_KEYS: Final[tuple[str, ...]] = (
    "raw_keystrokes",
    "raw_keys",
    "window_title_raw",
    "window_title",
    "clipboard_content",
    "typed_text",
    "im_content",
    "full_url",
)

_REDACTED: Final[str] = "[REDACTED]"

_SENSITIVE_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"(?P<key>"
    + "|".join(re.escape(k) for k in _SENSITIVE_KEYS)
    + r")\s*[=:]\s*(?P<value>\"[^\"]*\"|'[^']*'|\S+)",
    re.IGNORECASE,
)


def redact_message(message: str) -> str:
    """Replace sensitive ``key=value`` or ``key: value`` pairs with redaction markers.

    Args:
        message: Raw log message string.

    Returns:
        Message with sensitive values replaced by ``[REDACTED]``.
    """
    return _SENSITIVE_PATTERN.sub(
        lambda m: f"{m.group('key')}={_REDACTED}",
        message,
    )


@dataclass(eq=False)
class SanitizingFilter(logging.Filter):
    """A :class:`logging.Filter` that rewrites log records to strip sensitive data.

    Attach to any logger or handler via :func:`install_sanitizing_filter`
    to ensure sensitive key/value pairs never reach log output.
    """

    name: str = ""

    def __post_init__(self) -> None:
        super().__init__(self.name)

    def filter(self, record: logging.LogRecord) -> bool:
        if record.args:
            record.msg = redact_message(record.getMessage())
            record.args = None
        else:
            record.msg = redact_message(str(record.msg))
        return True


def install_sanitizing_filter(
    logger: logging.Logger | None = None,
    *,
    handler_level: bool = False,
) -> SanitizingFilter:
    """Attach a :class:`SanitizingFilter` to *logger* (or the root logger).

    Args:
        logger: Target logger.  Defaults to the root logger if ``None``.
        handler_level: If ``True``, install on each handler of *logger*
            instead of the logger itself.

    Returns:
        The filter instance that was installed (useful for later removal).
    """
    filt = SanitizingFilter()
    target = logger or logging.getLogger()

    if handler_level:
        for handler in target.handlers:
            handler.addFilter(filt)
    else:
        target.addFilter(filt)

    return filt


_FILE_LOG_FORMAT: Final[str] = (
    "%(asctime)s %(levelname)s %(name)s %(pathname)s:%(lineno)d — %(message)s"
)


def setup_file_logging(
    log_dir: str | Path | None = None,
    *,
    max_bytes: int = 5_000_000,
    backup_count: int = 3,
) -> logging.handlers.RotatingFileHandler | None:
    """Attach a :class:`~logging.handlers.RotatingFileHandler` to the root logger.

    The handler always logs at ``DEBUG`` level so that error context is
    captured even when the console level is ``WARNING``.  A
    :class:`SanitizingFilter` is applied so PII is never written to disk.

    If a ``RotatingFileHandler`` targeting the same file is already
    present on the root logger, this function is a no-op (idempotent).

    Args:
        log_dir: Directory for the log file.  Defaults to
            ``<TASKCLF_HOME>/logs`` when ``None``.
        max_bytes: Maximum size of a single log file before rotation.
        backup_count: Number of rotated backup files to keep.

    Returns:
        The handler that was created, or ``None`` if one already existed.
    """
    if log_dir is None:
        from taskclf.core.paths import taskclf_home

        log_dir = taskclf_home() / "logs"

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "taskclf.log"

    root = logging.getLogger()
    for h in root.handlers:
        if (
            isinstance(h, logging.handlers.RotatingFileHandler)
            and Path(h.baseFilename).resolve() == log_file.resolve()
        ):
            return None

    handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter(_FILE_LOG_FORMAT))
    handler.addFilter(SanitizingFilter())
    root.addHandler(handler)
    return handler
