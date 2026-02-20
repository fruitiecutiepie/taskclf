"""Security and privacy tests.

Covers: TC-SEC-001 (forbidden columns in artifacts), TC-SEC-002 (one-way hashing),
TC-SEC-003 (log sanitization — skipped).
"""

from __future__ import annotations

import datetime as dt
import inspect
import re

import pytest

import taskclf.core.hashing as hashing_mod
from taskclf.core.hashing import _HASH_TRUNCATION, salted_hash, stable_hash
from taskclf.core.store import read_parquet, write_parquet
from taskclf.features.build import generate_dummy_features

_FORBIDDEN_COLUMNS = frozenset({
    "raw_keystrokes",
    "raw_keys",
    "window_title_raw",
    "clipboard_content",
    "typed_text",
    "im_content",
    "full_url",
})


class TestForbiddenColumnsInArtifacts:
    """TC-SEC-001: persisted parquet must not contain forbidden columns."""

    def test_feature_parquet_has_no_raw_prefixed_columns(self, tmp_path) -> None:
        rows = generate_dummy_features(dt.date(2025, 6, 15), n_rows=10)
        import pandas as pd
        df = pd.DataFrame([r.model_dump() for r in rows])

        path = write_parquet(df, tmp_path / "features.parquet")
        loaded = read_parquet(path)

        raw_cols = [c for c in loaded.columns if c.startswith("raw_")]
        assert raw_cols == [], f"Forbidden raw_* columns found: {raw_cols}"

    def test_feature_parquet_excludes_explicit_forbidden_names(self, tmp_path) -> None:
        rows = generate_dummy_features(dt.date(2025, 6, 15), n_rows=10)
        import pandas as pd
        df = pd.DataFrame([r.model_dump() for r in rows])

        path = write_parquet(df, tmp_path / "features.parquet")
        loaded = read_parquet(path)

        present = set(loaded.columns) & _FORBIDDEN_COLUMNS
        assert present == set(), f"Forbidden columns found in artifact: {present}"

    def test_window_title_column_is_hashed_not_raw(self, tmp_path) -> None:
        rows = generate_dummy_features(dt.date(2025, 6, 15), n_rows=10)
        import pandas as pd
        df = pd.DataFrame([r.model_dump() for r in rows])

        assert "window_title_hash" in df.columns
        assert "window_title" not in df.columns
        assert "window_title_raw" not in df.columns

        for val in df["window_title_hash"]:
            assert re.fullmatch(r"[0-9a-f]+", val), (
                f"window_title_hash value {val!r} is not hex — may be raw text"
            )


class TestHashOneWay:
    """TC-SEC-002: hashing functions are one-way (SHA-256, no reverse API)."""

    def test_no_reverse_or_decode_function_exposed(self) -> None:
        public_names = [
            name for name in dir(hashing_mod)
            if not name.startswith("_") and callable(getattr(hashing_mod, name))
        ]
        suspect = {"decode", "reverse", "unhash", "decrypt", "invert"}
        exposed_suspect = suspect & set(public_names)
        assert exposed_suspect == set(), (
            f"Hashing module exposes reversibility API: {exposed_suspect}"
        )

    def test_hash_output_differs_from_plaintext(self) -> None:
        payload = "My Secret Document - Editor"
        assert salted_hash(payload, "salt") != payload
        assert stable_hash(payload) != payload

    def test_hash_output_is_fixed_length_hex(self) -> None:
        for payload in ("short", "a" * 1000, "", "unicode: Ωüñ"):
            result = salted_hash(payload, "salt")
            assert len(result) == _HASH_TRUNCATION
            assert re.fullmatch(r"[0-9a-f]+", result)

    def test_uses_sha256_not_reversible_encoding(self) -> None:
        source = inspect.getsource(hashing_mod)
        assert "sha256" in source.lower()
        for weak in ("base64", "rot13", "xor", "encode"):
            assert weak not in source.lower() or weak == "encode", (
                f"Hashing module uses potentially reversible method: {weak}"
            )


class TestLogSanitization:
    """TC-SEC-003: logs do not print sensitive raw payloads."""

    def test_redacts_window_title_value(self) -> None:
        from taskclf.core.logging import redact_message

        msg = 'Processing event window_title="My Secret Document - Editor"'
        result = redact_message(msg)
        assert "My Secret Document" not in result
        assert "window_title=[REDACTED]" in result

    def test_redacts_raw_keystrokes(self) -> None:
        from taskclf.core.logging import redact_message

        msg = "Captured raw_keystrokes=asdfghjkl in buffer"
        result = redact_message(msg)
        assert "asdfghjkl" not in result
        assert "raw_keystrokes=[REDACTED]" in result

    def test_redacts_clipboard_content(self) -> None:
        from taskclf.core.logging import redact_message

        msg = "clipboard_content='password123' pasted"
        result = redact_message(msg)
        assert "password123" not in result
        assert "clipboard_content=[REDACTED]" in result

    def test_preserves_safe_messages(self) -> None:
        from taskclf.core.logging import redact_message

        msg = "Processed 42 buckets for date=2025-06-15 in 0.3s"
        assert redact_message(msg) == msg

    def test_filter_on_real_logger(self) -> None:
        import logging

        from taskclf.core.logging import install_sanitizing_filter

        logger = logging.getLogger("taskclf.test.sanitize")
        logger.setLevel(logging.DEBUG)
        logger.handlers.clear()
        logger.propagate = False

        handler = logging.StreamHandler()
        handler.setLevel(logging.DEBUG)
        logger.addHandler(handler)

        filt = install_sanitizing_filter(logger)
        try:
            record = logger.makeRecord(
                "taskclf.test", logging.INFO, "", 0,
                'Event window_title="Top Secret" processed', (), None,
            )
            filt.filter(record)
            assert "Top Secret" not in record.getMessage()
            assert "[REDACTED]" in record.getMessage()
        finally:
            logger.removeFilter(filt)
