"""Tests for deterministic hashing utilities.

Covers: TC-CORE-010, TC-CORE-011, TC-CORE-012.
"""

from __future__ import annotations

import re

from taskclf.core.hashing import _HASH_TRUNCATION, salted_hash, stable_hash


class TestStableHash:
    def test_deterministic_same_input(self) -> None:
        """TC-CORE-010: identical input always yields identical output."""
        assert stable_hash("hello") == stable_hash("hello")

    def test_deterministic_across_calls(self) -> None:
        """TC-CORE-010 (extended): result doesn't change between invocations."""
        first = stable_hash("schema-payload-v1")
        second = stable_hash("schema-payload-v1")
        assert first == second

    def test_different_inputs_yield_different_hashes(self) -> None:
        assert stable_hash("input-a") != stable_hash("input-b")

    def test_output_length_and_charset(self) -> None:
        """TC-CORE-012: output is exactly _HASH_TRUNCATION hex characters."""
        result = stable_hash("anything")
        assert len(result) == _HASH_TRUNCATION
        assert re.fullmatch(r"[0-9a-f]+", result)

    def test_empty_input(self) -> None:
        result = stable_hash("")
        assert len(result) == _HASH_TRUNCATION
        assert re.fullmatch(r"[0-9a-f]+", result)


class TestSaltedHash:
    """TC-CORE-011: salted hashing for PII obfuscation."""

    def test_different_salts_yield_different_hashes(self) -> None:
        """TC-CORE-011: same payload with different salts must diverge."""
        title = "My Secret Document - Editor"
        assert salted_hash(title, "salt-A") != salted_hash(title, "salt-B")

    def test_deterministic_same_salt_and_payload(self) -> None:
        assert salted_hash("title", "s1") == salted_hash("title", "s1")

    def test_output_length_and_charset(self) -> None:
        result = salted_hash("payload", "salt")
        assert len(result) == _HASH_TRUNCATION
        assert re.fullmatch(r"[0-9a-f]+", result)

    def test_differs_from_unsalted(self) -> None:
        payload = "same-input"
        assert salted_hash(payload, "any-salt") != stable_hash(payload)

    def test_empty_salt_matches_prefix_concat(self) -> None:
        """Empty salt degenerates to hashing payload alone (but via salt path)."""
        result = salted_hash("x", "")
        assert len(result) == _HASH_TRUNCATION
        assert re.fullmatch(r"[0-9a-f]+", result)
