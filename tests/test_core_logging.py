"""Tests for sanitizing log filter (core/logging.py).

Covers: TC-LOG-001 (redact_message), TC-LOG-002 (SanitizingFilter),
TC-LOG-003 (install_sanitizing_filter).
"""

from __future__ import annotations

import logging

from taskclf.core.logging import (
    SanitizingFilter,
    _SENSITIVE_KEYS,
    install_sanitizing_filter,
    redact_message,
)


# ---------------------------------------------------------------------------
# redact_message
# ---------------------------------------------------------------------------


class TestRedactMessage:
    """TC-LOG-001: redact_message strips sensitive key/value pairs."""

    def test_redacts_key_equals_value(self) -> None:
        msg = "received raw_keystrokes=hello_world from input"
        result = redact_message(msg)
        assert "hello_world" not in result
        assert "raw_keystrokes=[REDACTED]" in result

    def test_redacts_key_colon_value(self) -> None:
        msg = "window_title: My Secret Document.docx"
        result = redact_message(msg)
        assert "My Secret Document.docx" not in result
        assert "window_title=[REDACTED]" in result

    def test_redacts_quoted_value(self) -> None:
        msg = 'clipboard_content="some pasted text here"'
        result = redact_message(msg)
        assert "some pasted text here" not in result
        assert "clipboard_content=[REDACTED]" in result

    def test_all_sensitive_keys_redacted(self) -> None:
        for key in _SENSITIVE_KEYS:
            msg = f"{key}=secret_value_123"
            result = redact_message(msg)
            assert "secret_value_123" not in result, f"Key {key!r} was not redacted"
            assert f"{key}=[REDACTED]" in result

    def test_non_sensitive_keys_pass_through(self) -> None:
        msg = "user_id=u-001 session_id=s-002"
        result = redact_message(msg)
        assert result == msg

    def test_multiple_sensitive_keys_in_one_message(self) -> None:
        msg = "raw_keystrokes=abc full_url=https://example.com"
        result = redact_message(msg)
        assert "abc" not in result
        assert "https://example.com" not in result
        assert result.count("[REDACTED]") == 2

    def test_case_insensitive(self) -> None:
        msg = "RAW_KEYSTROKES=secret"
        result = redact_message(msg)
        assert "secret" not in result
        assert "[REDACTED]" in result

    def test_empty_message(self) -> None:
        assert redact_message("") == ""

    def test_message_without_key_value_pairs(self) -> None:
        msg = "Just a regular log message with no secrets"
        assert redact_message(msg) == msg


# ---------------------------------------------------------------------------
# SanitizingFilter
# ---------------------------------------------------------------------------


class TestSanitizingFilter:
    """TC-LOG-002: SanitizingFilter rewrites log records."""

    def test_filter_always_returns_true(self) -> None:
        filt = SanitizingFilter()
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg="raw_keystrokes=secret", args=None, exc_info=None,
        )
        assert filt.filter(record) is True

    def test_redacts_msg_without_args(self) -> None:
        filt = SanitizingFilter()
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg="window_title=MyDoc.txt", args=None, exc_info=None,
        )
        filt.filter(record)
        assert "MyDoc.txt" not in record.msg
        assert "[REDACTED]" in record.msg

    def test_redacts_formatted_msg_with_args(self) -> None:
        filt = SanitizingFilter()
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg="Received %s from input",
            args=("raw_keystrokes=typing_data",), exc_info=None,
        )
        filt.filter(record)
        assert "typing_data" not in record.msg
        assert "[REDACTED]" in record.msg
        assert record.args is None


# ---------------------------------------------------------------------------
# install_sanitizing_filter
# ---------------------------------------------------------------------------


class TestInstallSanitizingFilter:
    """TC-LOG-003: install_sanitizing_filter attaches filter correctly."""

    def test_attaches_to_given_logger(self) -> None:
        logger = logging.getLogger("test.install.logger")
        logger.filters.clear()
        filt = install_sanitizing_filter(logger)
        assert isinstance(filt, SanitizingFilter)
        assert filt in logger.filters
        logger.removeFilter(filt)

    def test_attaches_to_root_logger_when_none(self) -> None:
        root = logging.getLogger()
        initial_count = len(root.filters)
        filt = install_sanitizing_filter(None)
        assert filt in root.filters
        root.removeFilter(filt)
        assert len(root.filters) == initial_count

    def test_handler_level_attaches_to_handlers(self) -> None:
        logger = logging.getLogger("test.install.handler")
        handler = logging.StreamHandler()
        logger.addHandler(handler)
        handler.filters.clear()

        filt = install_sanitizing_filter(logger, handler_level=True)
        assert filt in handler.filters
        assert filt not in logger.filters

        handler.removeFilter(filt)
        logger.removeHandler(handler)
