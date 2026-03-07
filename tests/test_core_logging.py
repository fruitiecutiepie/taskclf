"""Tests for sanitizing log filter (core/logging.py).

Covers: TC-LOG-001 (redact_message), TC-LOG-002 (SanitizingFilter),
TC-LOG-003 (install_sanitizing_filter), TC-LOG-004 (setup_file_logging).
"""

from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path

import pytest

from taskclf.core.logging import (
    SanitizingFilter,
    _SENSITIVE_KEYS,
    install_sanitizing_filter,
    redact_message,
    setup_file_logging,
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


# ---------------------------------------------------------------------------
# setup_file_logging
# ---------------------------------------------------------------------------


class TestSetupFileLogging:
    """TC-LOG-004: setup_file_logging creates a RotatingFileHandler with sanitization."""

    def _cleanup_handler(self, handler: logging.handlers.RotatingFileHandler | None) -> None:
        if handler is not None:
            logging.getLogger().removeHandler(handler)
            handler.close()

    def test_creates_log_file(self, tmp_path: Path) -> None:
        log_dir = tmp_path / "logs"
        handler = setup_file_logging(log_dir)
        try:
            assert handler is not None
            assert (log_dir / "taskclf.log").exists()
        finally:
            self._cleanup_handler(handler)

    def test_creates_log_directory(self, tmp_path: Path) -> None:
        log_dir = tmp_path / "deep" / "nested" / "logs"
        handler = setup_file_logging(log_dir)
        try:
            assert log_dir.is_dir()
        finally:
            self._cleanup_handler(handler)

    def test_handler_level_is_debug(self, tmp_path: Path) -> None:
        handler = setup_file_logging(tmp_path / "logs")
        try:
            assert handler is not None
            assert handler.level == logging.DEBUG
        finally:
            self._cleanup_handler(handler)

    def test_handler_has_sanitizing_filter(self, tmp_path: Path) -> None:
        handler = setup_file_logging(tmp_path / "logs")
        try:
            assert handler is not None
            assert any(isinstance(f, SanitizingFilter) for f in handler.filters)
        finally:
            self._cleanup_handler(handler)

    def test_handler_attached_to_root_logger(self, tmp_path: Path) -> None:
        handler = setup_file_logging(tmp_path / "logs")
        try:
            assert handler is not None
            assert handler in logging.getLogger().handlers
        finally:
            self._cleanup_handler(handler)

    def test_idempotent_returns_none_on_second_call(self, tmp_path: Path) -> None:
        log_dir = tmp_path / "logs"
        handler1 = setup_file_logging(log_dir)
        try:
            handler2 = setup_file_logging(log_dir)
            assert handler2 is None
        finally:
            self._cleanup_handler(handler1)

    def test_respects_rotation_params(self, tmp_path: Path) -> None:
        handler = setup_file_logging(
            tmp_path / "logs", max_bytes=1_000_000, backup_count=5,
        )
        try:
            assert handler is not None
            assert handler.maxBytes == 1_000_000
            assert handler.backupCount == 5
        finally:
            self._cleanup_handler(handler)

    def test_sanitizes_sensitive_data_in_file(self, tmp_path: Path) -> None:
        log_dir = tmp_path / "logs"
        handler = setup_file_logging(log_dir)
        try:
            assert handler is not None
            test_logger = logging.getLogger("test.file_sanitize")
            test_logger.setLevel(logging.DEBUG)
            test_logger.debug("window_title=SecretDoc.txt")
            handler.flush()

            content = (log_dir / "taskclf.log").read_text()
            assert "SecretDoc.txt" not in content
            assert "[REDACTED]" in content
        finally:
            self._cleanup_handler(handler)

    def test_default_log_dir_uses_taskclf_home(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("TASKCLF_HOME", str(tmp_path / "home"))
        handler = setup_file_logging(None)
        try:
            assert handler is not None
            assert Path(handler.baseFilename).parent == tmp_path / "home" / "logs"
        finally:
            self._cleanup_handler(handler)
