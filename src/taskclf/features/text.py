"""Window-title featurization using hash-based approaches.

All functions operate on hashed or salted representations — raw window
titles are never stored or returned.  This satisfies the project's
privacy invariant (see ``AGENTS.md``).
"""

from __future__ import annotations

from taskclf.core.hashing import salted_hash


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
    to Python's built-in ``hash()`` for deterministic bucketing.

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
        return abs(hash(title_hash)) % n_buckets
