from __future__ import annotations

# TODO: salt handling (salted hashing for window titles, separate from schema hashing)

import hashlib

_HASH_TRUNCATION = 12

def stable_hash(payload: str) -> str:
    """Deterministic SHA-256 of *payload*, truncated to 12 hex chars.

    Used for schema hashing (not for title hashing — that uses a salted
    variant handled elsewhere).
    """
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return digest[:_HASH_TRUNCATION]
