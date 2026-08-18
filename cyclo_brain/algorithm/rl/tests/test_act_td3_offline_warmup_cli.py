"""Focused tests for the standalone ACT-TD3 warm-up command contract."""

from __future__ import annotations

import argparse
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path

from cyclo_brain.algorithm.rl.act_td3 import offline_warmup_cli as cli


def _arguments(root: Path) -> list[str]:
    return [
        "--dataset-root",
        str(root / "dataset"),
        "--act-checkpoint",
        str(root / "actor"),
        "--robot-config",
        str(root / "robot.yaml"),
        "--robot-type",
        "ffw_sg2_rev1",
        "--device",
        "cuda:0",
        "--seed",
        "17",
        "--batch-size",
        "4",
        "--checkpoint",
        str(root / "output" / "latest.pt"),
    ]


class ACTTD3OfflineWarmupCLITest(unittest.TestCase):
    def test_parser_keeps_algorithm_boundary_separate_from_sampling_seed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = cli.build_parser().parse_args(
                [
                    *_arguments(root),
                    "--sampling-seed",
                    "19",
                    "--max-critic-updates",
                    "1",
                ]
            )

        self.assertEqual(args.seed, 17)
        self.assertEqual(args.sampling_seed, 19)
        self.assertEqual(args.max_critic_updates, 1)
        self.assertEqual(args.checkpoint_interval, 500)
        self.assertEqual(args.progress_interval, 10)
        self.assertEqual(args.video_backend, "pyav")

    def test_parser_rejects_boundary_past_fixed_warmup(self) -> None:
        parser = cli.build_parser()
        with tempfile.TemporaryDirectory() as directory:
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parser.parse_args(
                    [
                        *_arguments(Path(directory)),
                        "--max-critic-updates",
                        "5001",
                    ]
                )

    def test_checkpoint_cannot_overwrite_or_live_inside_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset"
            actor = root / "actor"
            dataset.mkdir()
            actor.mkdir()
            robot = root / "config" / "robot.yaml"
            robot.parent.mkdir()
            robot.write_text("robot", encoding="utf-8")
            existing = root / "existing.pt"
            existing.write_bytes(b"state")

            with self.assertRaises(FileExistsError):
                cli._output_checkpoint(
                    existing,
                    resume=False,
                    inputs=(dataset, actor, robot),
                )
            with self.assertRaisesRegex(ValueError, "outside"):
                cli._output_checkpoint(
                    dataset / "latest.pt",
                    resume=False,
                    inputs=(dataset, actor, robot),
                )
            self.assertEqual(
                cli._output_checkpoint(
                    existing,
                    resume=True,
                    inputs=(dataset, actor, robot),
                ),
                existing.resolve(),
            )

    def test_json_lines_are_strict_and_machine_readable(self) -> None:
        stream = io.StringIO()
        cli._json_line({"event": "progress", "percentage": 0.02}, stream=stream)
        self.assertEqual(
            json.loads(stream.getvalue()),
            {"event": "progress", "percentage": 0.02},
        )
        with self.assertRaises(ValueError):
            cli._json_line({"bad": float("nan")}, stream=stream)

    def test_device_contract_requires_explicit_cuda_index(self) -> None:
        self.assertEqual(str(cli._device("cpu")), "cpu")
        with self.assertRaisesRegex(ValueError, "explicit CUDA index"):
            cli._device("cuda")

    def test_required_arguments_are_not_silently_defaulted(self) -> None:
        parser = cli.build_parser()
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parser.parse_args([])
        self.assertIsInstance(parser, argparse.ArgumentParser)


if __name__ == "__main__":
    unittest.main()
