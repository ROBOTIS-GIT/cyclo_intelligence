"""Stable JSON fingerprints and SHA-256 digest validation.

The canonical JSON encoding is part of the persisted learning-artifact
contract.  Keep its options explicit so every producer and verifier hashes
the exact same bytes.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical_json_sha256(value: Any, *, allow_nan: bool = True) -> str:
    """Return SHA-256 for Cyclo's canonical UTF-8 JSON representation."""

    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=allow_nan,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_lowercase_sha256(value: Any, *, error_message: str) -> str:
    """Return a valid lowercase SHA-256 digest or raise the caller's error."""

    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(error_message)
    return value


__all__ = ["canonical_json_sha256", "validate_lowercase_sha256"]
