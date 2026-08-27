"""Focused tests for the standalone ACT-TD3 warm-up command contract."""

from __future__ import annotations

import argparse
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

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
                    "--critic-updates",
                    "1200",
                    "--max-critic-updates",
                    "1200",
                ]
            )

        self.assertEqual(args.seed, 17)
        self.assertEqual(args.dataset_root, [root / "dataset"])
        self.assertEqual(args.sampling_seed, 19)
        self.assertEqual(args.critic_updates, 1200)
        self.assertEqual(args.max_critic_updates, 1200)
        self.assertEqual(args.checkpoint_interval, 500)
        self.assertEqual(args.progress_interval, 10)
        self.assertEqual(args.video_backend, "pyav")
        self.assertIsNone(args.publish_dir)

    def test_parser_rejects_update_count_past_supported_limit(self) -> None:
        parser = cli.build_parser()
        with tempfile.TemporaryDirectory() as directory:
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parser.parse_args(
                    [
                        *_arguments(Path(directory)),
                        "--critic-updates",
                        "1000001",
                    ]
                )

    def test_run_rejects_boundary_past_selected_warmup(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot exceed"):
            cli.run_from_args(SimpleNamespace(
                critic_updates=1200,
                max_critic_updates=1201,
            ))

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

    def test_publish_directory_is_exactly_policy_local_critic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            actor = Path(directory) / "actor"
            actor.mkdir()

            self.assertEqual(
                cli._publish_directory(  # noqa: SLF001
                    actor / "critic",
                    act_checkpoint=actor.resolve(),
                ),
                actor.resolve() / "critic",
            )
            with self.assertRaisesRegex(ValueError, "exactly"):
                cli._publish_directory(  # noqa: SLF001
                    actor / "somewhere-else",
                    act_checkpoint=actor.resolve(),
                )

            outside = Path(directory) / "outside"
            (actor / "critic").symlink_to(outside, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "symbolic link"):
                cli._publish_directory(  # noqa: SLF001
                    actor / "critic",
                    act_checkpoint=actor.resolve(),
                )

    def test_policy_local_runner_rejects_symlinked_runs_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            actor = root / "actor"
            critic = actor / "critic"
            outside = root / "outside"
            actor.mkdir()
            critic.mkdir()
            outside.mkdir()
            (critic / "runs").symlink_to(outside, target_is_directory=True)

            with self.assertRaisesRegex(ValueError, "symbolic link"):
                cli._output_checkpoint(  # noqa: SLF001
                    critic / "runs" / "job.pt",
                    resume=False,
                    inputs=(actor,),
                    allowed_output_root=critic / "runs",
                )

    @staticmethod
    def _fake_publish_runner() -> SimpleNamespace:
        artifact = {
            "format": "cyclo_brain.act_td3_critic/v1",
            "status": "complete",
            "contract": {
                "dataset": {"episode_count": 2},
                "learner": {"action_dim": 2},
            },
            "actor_sha256": "a" * 64,
            "actor_target_sha256": "a" * 64,
            "critic": {"weight": torch.tensor([1.0])},
            "critic_target": {"weight": torch.tensor([2.0])},
            "critic_optimizer": {"state": {}, "param_groups": []},
            "completed_critic_updates": 4,
            "completed_actor_updates": 0,
        }
        return SimpleNamespace(
            CRITIC_ARTIFACT_FORMAT="cyclo_brain.act_td3_critic/v1",
            total_critic_updates=4,
            critic_artifact_state=lambda: artifact,
        )

    def test_completed_critic_publish_is_verified_and_writes_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            actor = root / "actor"
            actor.mkdir()
            identity = SimpleNamespace(
                identity="sha256:data",
                file_count=3,
                byte_count=42,
                component_sha256={"dataset": "sha256:dataset"},
                virtual_contract={"data_roots": []},
            )

            runner = self._fake_publish_runner()
            real_contract = runner.critic_artifact_state()["contract"]
            real_contract["learner"]["observation_keys"] = (
                "observation.images.front",
                "observation.state",
            )
            real_contract["learner"]["config"] = {
                "actor_trainable_groups": ("action_decoder",),
            }
            checkpoint, manifest_path = cli._publish_completed_critic(  # noqa: SLF001
                runner=runner,
                publish_dir=actor / "critic",
                act_checkpoint=actor,
                dataset_roots=(root / "data-0", root / "data-1"),
                identity=identity,
            )

            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(state["format"], "cyclo_brain.act_td3_critic/v1")
            self.assertNotIn("actor", state)
            self.assertEqual(
                manifest["format"],
                "cyclo_brain.act_td3_critic_manifest/v1",
            )
            self.assertTrue(manifest["actor_exactly_unchanged"])
            self.assertEqual(manifest["artifact"]["checkpoint_path"], "latest.pt")
            self.assertEqual(
                manifest["learner"]["observation_keys"],
                ["observation.images.front", "observation.state"],
            )
            self.assertEqual(
                manifest["learner"]["config"]["actor_trainable_groups"],
                ["action_decoder"],
            )
            self.assertEqual(
                manifest["artifact"]["sha256"],
                cli._sha256_file(checkpoint),  # noqa: SLF001
            )

    def test_failed_publish_restores_previous_latest_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            actor = root / "actor"
            publish = actor / "critic"
            publish.mkdir(parents=True)
            latest = publish / "latest.pt"
            manifest_path = publish / "manifest.json"
            latest.write_bytes(b"previous critic")
            manifest_path.write_bytes(b"previous manifest")
            previous_latest = latest.read_bytes()
            previous_manifest = manifest_path.read_bytes()
            identity = SimpleNamespace(
                identity="sha256:data",
                file_count=1,
                byte_count=1,
                component_sha256={},
                virtual_contract={},
            )
            original_replace = cli.os.replace

            def fail_manifest_commit(source, destination):
                if Path(destination) == manifest_path:
                    raise OSError("manifest commit failed")
                return original_replace(source, destination)

            with mock.patch.object(
                cli.os,
                "replace",
                side_effect=fail_manifest_commit,
            ):
                with self.assertRaisesRegex(OSError, "manifest commit failed"):
                    cli._publish_completed_critic(  # noqa: SLF001
                        runner=self._fake_publish_runner(),
                        publish_dir=publish,
                        act_checkpoint=actor,
                        dataset_roots=(root / "data",),
                        identity=identity,
                    )

            self.assertEqual(latest.read_bytes(), previous_latest)
            self.assertEqual(manifest_path.read_bytes(), previous_manifest)
            self.assertFalse(any(publish.glob(".*.prepared")))
            self.assertFalse(any(publish.glob(".*.backup")))

    def test_changed_identity_is_rejected_before_publish(self) -> None:
        expected = SimpleNamespace(identity="sha256:before")
        changed = SimpleNamespace(identity="sha256:after")
        with mock.patch.object(
            cli,
            "build_act_td3_multi_root_training_data_identity",
            return_value=changed,
        ) as rebuild:
            with self.assertRaisesRegex(RuntimeError, "changed during"):
                cli._require_unchanged_training_data_identity(  # noqa: SLF001
                    expected=expected,
                    datasets=(object(),),
                    dataset_roots=(Path("/dataset"),),
                    act_checkpoint=Path("/actor"),
                    action_domains=(object(),),
                    robot_type="ffw_sg2_rev1",
                    video_backend="pyav",
                )
        rebuild.assert_called_once()

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

    def test_main_turns_signals_into_cooperative_stop(self) -> None:
        installed: dict[int, object] = {}
        previous = {
            cli.signal.SIGINT: object(),
            cli.signal.SIGTERM: object(),
        }

        def fake_signal(signum, handler):
            installed[signum] = handler

        def fake_run(_args, *, should_stop):
            self.assertFalse(should_stop())
            installed[cli.signal.SIGINT](cli.signal.SIGINT, None)
            self.assertTrue(should_stop())
            return object()

        parser = SimpleNamespace(parse_args=lambda _argv: SimpleNamespace())
        with (
            mock.patch.object(cli, "build_parser", return_value=parser),
            mock.patch.object(
                cli.signal,
                "getsignal",
                side_effect=lambda signum: previous[signum],
            ),
            mock.patch.object(cli.signal, "signal", side_effect=fake_signal),
            mock.patch.object(cli, "run_from_args", side_effect=fake_run),
        ):
            self.assertEqual(cli.main([]), 0)

        self.assertIs(installed[cli.signal.SIGINT], previous[cli.signal.SIGINT])
        self.assertIs(installed[cli.signal.SIGTERM], previous[cli.signal.SIGTERM])


if __name__ == "__main__":
    unittest.main()
