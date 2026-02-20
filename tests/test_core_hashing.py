"""Tests for deterministic hashing utilities.

Covers: TC-CORE-010, TC-CORE-012.  TC-CORE-011 skipped (salted hashing not yet implemented).
"""

from __future__ import annotations

import re

import pytest

from taskclf.core.hashing import _HASH_TRUNCATION, stable_hash


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


@pytest.mark.skip(reason="TODO: remove .skip once salted hashing is implemented in core/hashing.py")
class TestSaltedHash:
    """TC-CORE-011: different salts yield different hashes (salted variant)."""

    def test_different_salts_yield_different_hashes(self) -> None:
        raise NotImplementedError
