#!/usr/bin/env python3

from __future__ import annotations

import argparse
import io
from pathlib import Path
import sys
import tempfile
import unittest


GROOT_ROOT = Path(__file__).resolve().parents[1]
if str(GROOT_ROOT) not in sys.path:
    sys.path.insert(0, str(GROOT_ROOT))

from runtime.rlt_cli_common import (  # noqa: E402
    json_line,
    positive_int,
    prepare_output_directory,
    resolved_directory,
)


class RLTCLICommonTests(unittest.TestCase):
    def test_positive_int_contract(self) -> None:
        self.assertEqual(positive_int("7"), 7)
        with self.assertRaisesRegex(argparse.ArgumentTypeError, "must be positive"):
            positive_int("0")
        with self.assertRaisesRegex(argparse.ArgumentTypeError, "must be an integer"):
            positive_int("x")

    def test_json_line_keeps_compact_sorted_unicode_contract(self) -> None:
        stream = io.StringIO()
        json_line({"z": "젤리", "a": 1}, stream=stream)
        self.assertEqual(stream.getvalue(), '{"a":1,"z":"젤리"}\n')

    def test_directory_and_output_guards_preserve_stage_labels(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            self.assertEqual(resolved_directory(source, "source"), source.absolute())

            output = root / "output"
            self.assertEqual(
                prepare_output_directory(
                    output,
                    (source,),
                    stage_label="RLT Stage 1",
                    overlap_description="an input directory",
                ),
                output.absolute(),
            )
            with self.assertRaisesRegex(
                ValueError, "RLT Stage 2 output overlaps an input"
            ):
                prepare_output_directory(
                    source / "nested",
                    (source,),
                    stage_label="RLT Stage 2",
                    overlap_description="an input",
                )


if __name__ == "__main__":
    unittest.main()
