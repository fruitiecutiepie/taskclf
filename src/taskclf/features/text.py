"""Window-title featurization using hash-based approaches.

All functions operate on hashed or salted representations — raw window
titles are never stored or returned.  This satisfies the project's
privacy invariant (see ``AGENTS.md``).
"""

from __future__ import annotations

import re
import unicodedata
import hashlib
from dataclasses import dataclass

from taskclf.core.defaults import (
    DEFAULT_TITLE_CHAR3_SKETCH_BUCKETS,
    DEFAULT_TITLE_TOKEN_SKETCH_BUCKETS,
)
from taskclf.core.hashing import keyed_digest, salted_hash

_WHITESPACE_RE = re.compile(r"\s+")
_DIGIT_RUN_RE = re.compile(r"\d+")
_NON_ALNUM_SPLIT_RE = re.compile(r"[^0-9A-Za-z]+")
_SEPARATOR_CHARS = "|-:/•"
_MAX_TITLE_CHARS = 256
_MAX_TOKENS = 32
_TOKEN_NAMESPACE = "title_token_sketch_v1"
_CHAR3_NAMESPACE = "title_char3_sketch_v1"


@dataclass(frozen=True, slots=True)
class TitleSketchFeatures:
    """Privacy-safe title features derived from a raw window title."""

    title_token_sketch: tuple[float, ...]
    title_char3_sketch: tuple[float, ...]
    title_char_count: int
    title_token_count: int
    title_unique_token_ratio: float
    title_digit_ratio: float
    title_separator_count: int


def normalize_title(raw_title: str) -> str:
    """Normalize a raw window title before featurization."""
    normalized = unicodedata.normalize("NFKC", raw_title)
    normalized = _WHITESPACE_RE.sub(" ", normalized).strip()
    return normalized


def _normalize_for_tokens(normalized_title: str) -> list[str]:
    lowered = normalized_title.lower()
    lowered = _DIGIT_RUN_RE.sub("<num>", lowered)
    parts = _NON_ALNUM_SPLIT_RE.split(lowered)
    tokens = [part for part in parts if part and (len(part) > 1 or part == "<num>")]
    return tokens[:_MAX_TOKENS]


def _normalize_for_char_ngrams(normalized_title: str) -> str:
    lowered = normalized_title.lower()
    lowered = _DIGIT_RUN_RE.sub("0", lowered)
    return lowered[:_MAX_TITLE_CHARS]


def _normalized_bucket_frequencies(
    items: list[str],
    *,
    secret: str,
    namespace: str,
    n_buckets: int,
) -> tuple[float, ...]:
    if n_buckets < 1:
        raise ValueError(f"n_buckets must be >= 1, got {n_buckets}")
    if not items:
        return tuple(0.0 for _ in range(n_buckets))

    bins = [0.0] * n_buckets
    for item in items:
        digest = keyed_digest(item, secret, namespace)
        bucket = int.from_bytes(digest[:8], "big") % n_buckets
        bins[bucket] += 1.0

    total = float(len(items))
    return tuple(round(count / total, 6) for count in bins)


def derive_title_sketch_features(
    raw_title: str,
    secret: str,
    *,
    token_buckets: int = DEFAULT_TITLE_TOKEN_SKETCH_BUCKETS,
    char3_buckets: int = DEFAULT_TITLE_CHAR3_SKETCH_BUCKETS,
) -> TitleSketchFeatures:
    """Convert a raw title into non-reversible keyed sketch features."""
    normalized = normalize_title(raw_title)
    token_inputs = _normalize_for_tokens(normalized)
    char_source = _normalize_for_char_ngrams(normalized)
    char3_inputs = [char_source[i : i + 3] for i in range(max(len(char_source) - 2, 0))]

    char_count = len(normalized)
    digit_chars = sum(ch.isdigit() for ch in normalized)
    separator_count = sum(ch in _SEPARATOR_CHARS for ch in normalized)
    unique_ratio = len(set(token_inputs)) / len(token_inputs) if token_inputs else 0.0
    digit_ratio = (digit_chars / char_count) if char_count > 0 else 0.0

    return TitleSketchFeatures(
        title_token_sketch=_normalized_bucket_frequencies(
            token_inputs,
            secret=secret,
            namespace=_TOKEN_NAMESPACE,
            n_buckets=token_buckets,
        ),
        title_char3_sketch=_normalized_bucket_frequencies(
            char3_inputs,
            secret=secret,
            namespace=_CHAR3_NAMESPACE,
            n_buckets=char3_buckets,
        ),
        title_char_count=char_count,
        title_token_count=len(token_inputs),
        title_unique_token_ratio=round(unique_ratio, 6),
        title_digit_ratio=round(digit_ratio, 6),
        title_separator_count=separator_count,
    )


def featurize_title(raw_title: str, salt: str) -> str:
    """Convert a raw window title into a privacy-safe salted hash.

    This is the single entry-point for title featurization.  Callers
    should discard the raw title immediately after calling this function.

    Args:
        raw_title: The original window title string.
        salt: A per-installation or per-session secret used to prevent
            rainbow-table attacks on the hash.

    Returns:
        A 12-character hex digest that is deterministic for the same
        *(raw_title, salt)* pair but infeasible to reverse.
    """
    return salted_hash(raw_title, salt)


def title_hash_bucket(title_hash: str, n_buckets: int = 256) -> int:
    """Map a title hash to an integer bucket index via the hash trick.

    Useful for converting the opaque hex hash into a bounded categorical
    feature that tree-based or embedding models can consume directly.

    If *title_hash* is not valid hex (e.g. from test data), falls back
    to a SHA-256 digest for deterministic bucketing.

    Args:
        title_hash: Hex string produced by :func:`featurize_title`
            (or :func:`~taskclf.core.hashing.salted_hash`).
        n_buckets: Size of the output space.  Must be >= 1.

    Returns:
        Integer in ``[0, n_buckets)``.

    Raises:
        ValueError: If *n_buckets* < 1.
    """
    if n_buckets < 1:
        raise ValueError(f"n_buckets must be >= 1, got {n_buckets}")
    try:
        return int(title_hash, 16) % n_buckets
    except ValueError:
        return (
            int.from_bytes(hashlib.sha256(title_hash.encode()).digest()[:8], "big")
            % n_buckets
        )
