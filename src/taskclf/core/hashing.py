"""Deterministic hashing utilities for schema fingerprinting and PII obfuscation."""

from __future__ import annotations

import hashlib

_HASH_TRUNCATION = 12


def stable_hash(payload: str) -> str:
    """Deterministic SHA-256 of *payload*, truncated to 12 hex chars.

    Used for schema hashing (not for title hashing -- that uses the salted
    variant :func:`salted_hash`).

    Args:
        payload: Arbitrary string to hash.

    Returns:
        First 12 hexadecimal characters of the SHA-256 digest.
    """
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return digest[:_HASH_TRUNCATION]


def salted_hash(payload: str, salt: str) -> str:
    """Salted SHA-256 of *payload*, truncated to 12 hex chars.

    Prepends *salt* to *payload* before hashing so the same input produces
    different digests under different salts.  Intended for window-title and
    other PII obfuscation where reversibility must be infeasible even if the
    hash function is known.

    Args:
        payload: Arbitrary string to hash (e.g. a window title).
        salt: A per-installation or per-session secret.

    Returns:
        First 12 hexadecimal characters of the SHA-256 digest.
    """
    digest = hashlib.sha256((salt + payload).encode("utf-8")).hexdigest()
    return digest[:_HASH_TRUNCATION]
