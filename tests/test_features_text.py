"""Tests for window-title featurization (features/text.py).

Covers:
- Hash-only output (no raw title leakage)
- Deterministic hashing
- Distinct titles produce distinct hashes
- title_hash_bucket returns a valid bucket index
"""

from __future__ import annotations

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
