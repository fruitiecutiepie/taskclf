"""Sanitizing log filter that redacts sensitive payloads before they reach handlers.

Prevents accidental leakage of raw window titles, keystrokes, clipboard
content, URLs, and other PII through log output.
"""

from __future__ import annotations

import logging
import re
from typing import Final, Sequence

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
        lambda m: f"{m.group('key')}={_REDACTED}", message,
    )


class SanitizingFilter(logging.Filter):
    """A :class:`logging.Filter` that rewrites log records to strip sensitive data.

    Attach to any logger or handler via :func:`install_sanitizing_filter`
    to ensure sensitive key/value pairs never reach log output.
    """

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
