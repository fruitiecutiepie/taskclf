"""Tests for window-title featurization (features/text.py).

Covers:
- Hash-only output (no raw title leakage)
- Deterministic hashing
- Distinct titles produce distinct hashes
- title_hash_bucket returns a valid bucket index
- title_hash_bucket edge cases (invalid n_buckets, non-hex, custom n_buckets)
- featurize_title edge cases (empty title, different salts)
"""

from __future__ import annotations

import pytest

from taskclf.features.text import featurize_title, title_hash_bucket


class TestFeaturizeTitle:
    def test_no_raw_title_in_output(self) -> None:
        """TC-FEAT-004: window title featurization uses hash/tokenization only."""
        raw_title = "my-secret-document.docx - Microsoft Word"
        salt = "test-salt-value"

        result = featurize_title(raw_title, salt)

        assert raw_title not in result
        for fragment in ("secret", "document", "Microsoft", "Word"):
            assert fragment not in result

        assert len(result) == 12
        assert all(c in "0123456789abcdef" for c in result)

    def test_deterministic(self) -> None:
        raw_title = "my-secret-document.docx - Microsoft Word"
        salt = "test-salt-value"
        assert featurize_title(raw_title, salt) == featurize_title(raw_title, salt)

    def test_distinct_titles_distinct_hashes(self) -> None:
        salt = "test-salt-value"
        result = featurize_title("my-secret-document.docx - Microsoft Word", salt)
        other = featurize_title("a-completely-different-title.txt", salt)
        assert other != result

    def test_hash_bucket(self) -> None:
        result = featurize_title("anything", "salt")
        bucket = title_hash_bucket(result)
        assert isinstance(bucket, int)
        assert 0 <= bucket < 256

    def test_empty_title(self) -> None:
        """TC-FEAT-TEXT-006: empty string title produces valid 12-char hex."""
        result = featurize_title("", "salt")
        assert len(result) == 12
        assert all(c in "0123456789abcdef" for c in result)

    def test_different_salts_different_hashes(self) -> None:
        """TC-FEAT-TEXT-007: same title with different salts produces different hashes."""
        title = "Untitled - Notepad"
        h1 = featurize_title(title, "salt-alpha")
        h2 = featurize_title(title, "salt-beta")
        assert h1 != h2


class TestTitleHashBucket:
    def test_n_buckets_zero_raises(self) -> None:
        """TC-FEAT-TEXT-001: n_buckets=0 raises ValueError."""
        with pytest.raises(ValueError, match="n_buckets must be >= 1"):
            title_hash_bucket("aabbccddee00", n_buckets=0)

    def test_n_buckets_negative_raises(self) -> None:
        """TC-FEAT-TEXT-002: n_buckets=-1 raises ValueError."""
        with pytest.raises(ValueError, match="n_buckets must be >= 1"):
            title_hash_bucket("aabbccddee00", n_buckets=-1)

    def test_non_hex_fallback(self) -> None:
        """TC-FEAT-TEXT-003: non-hex input falls back to hash(), returns valid int."""
        result = title_hash_bucket("not-hex-string", n_buckets=256)
        assert isinstance(result, int)
        assert 0 <= result < 256

    def test_custom_n_buckets(self) -> None:
        """TC-FEAT-TEXT-004: custom n_buckets=10 produces result in [0, 10)."""
        result = title_hash_bucket("aabbccddee00", n_buckets=10)
        assert isinstance(result, int)
        assert 0 <= result < 10

    def test_n_buckets_one(self) -> None:
        """TC-FEAT-TEXT-005: n_buckets=1 always returns 0."""
        assert title_hash_bucket("aabbccddee00", n_buckets=1) == 0
        assert title_hash_bucket("1234567890ab", n_buckets=1) == 0
        assert title_hash_bucket("not-hex", n_buckets=1) == 0
