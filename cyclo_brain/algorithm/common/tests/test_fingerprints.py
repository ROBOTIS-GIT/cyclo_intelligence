"""Regression tests for persisted JSON and digest contracts."""

from __future__ import annotations

import unittest

from cyclo_brain.algorithm.common.fingerprints import (
    canonical_json_sha256,
    validate_lowercase_sha256,
)


class FingerprintTest(unittest.TestCase):
    def test_canonical_json_sha256_preserves_existing_encoding(self) -> None:
        value = {"unicode": "로봇", "nested": {"z": 1, "a": True}}

        self.assertEqual(
            canonical_json_sha256(value),
            "d50f3e3d3c313e193d027d4d1cb136cdcbbc5fa17681bbcf12d1faf4cd2b717d",
        )
        self.assertEqual(
            canonical_json_sha256({"nested": {"a": True, "z": 1}, "unicode": "로봇"}),
            canonical_json_sha256(value),
        )

    def test_strict_json_fingerprint_rejects_non_finite_numbers(self) -> None:
        with self.assertRaisesRegex(ValueError, "Out of range float values"):
            canonical_json_sha256({"loss": float("nan")}, allow_nan=False)

    def test_digest_validation_preserves_the_callers_error(self) -> None:
        digest = "a" * 64
        self.assertIs(validate_lowercase_sha256(digest, error_message="unused"), digest)

        for invalid in (None, 1, "a" * 63, "A" * 64, "g" * 64):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "^exact contract error$"):
                    validate_lowercase_sha256(
                        invalid,
                        error_message="exact contract error",
                    )


if __name__ == "__main__":
    unittest.main()
