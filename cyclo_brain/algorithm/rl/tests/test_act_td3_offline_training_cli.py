"""Focused tests for the staged ACT-TD3 command and model export."""

from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
from torch import nn

from cyclo_brain.algorithm.rl.act_td3 import offline_training_cli as cli


class _FakePolicy(nn.Linear):
    def __init__(self) -> None:
        super().__init__(2, 2)

    def save_pretrained(self, directory: str | Path) -> None:
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        (root / "config.json").write_text("{}\n", encoding="utf-8")
        torch.save(self.state_dict(), root / "model.safetensors")


class _FakeProcessor:
    def __init__(self, filename: str) -> None:
        self.filename = filename

    def save_pretrained(
        self,
        directory: str | Path,
        *,
        config_filename: str,
    ) -> None:
        self.assert_filename(config_filename)
        (Path(directory) / config_filename).write_text("{}\n", encoding="utf-8")

    def assert_filename(self, value: str) -> None:
        if value != self.filename:
            raise AssertionError(f"unexpected processor filename: {value}")


class ACTTD3OfflineTrainingCLITest(unittest.TestCase):
    def test_parser_requires_one_output_and_rejects_two_continuations(self) -> None:
        parser = cli.build_parser()
        required = [
            "--dataset-root",
            "/dataset",
            "--act-checkpoint",
            "/actor",
            "--robot-config",
            "/robot.yaml",
            "--robot-type",
            "ffw_sg2_rev1",
            "--device",
            "cuda:0",
            "--seed",
            "17",
            "--batch-size",
            "4",
            "--output-dir",
            "/output",
        ]
        args = parser.parse_args(required)
        self.assertEqual(args.batch_size, 4)
        self.assertFalse(args.resume)
        self.assertIsNone(args.parent_checkpoint)
        self.assertIsNone(args.max_round_critic_updates)
        self.assertEqual(args.critic_epochs, 10)
        self.assertEqual(args.actor_equivalent_epochs, 5)
        self.assertFalse(args.allow_partial_round)

        configured = parser.parse_args(
            [
                *required,
                "--critic-epochs",
                "6",
                "--actor-equivalent-epochs",
                "3",
                "--allow-partial-round",
            ]
        )
        self.assertEqual(configured.critic_epochs, 6)
        self.assertEqual(configured.actor_equivalent_epochs, 3)
        self.assertTrue(configured.allow_partial_round)

        with self.assertRaises(SystemExit):
            parser.parse_args(
                [*required, "--resume", "--parent-checkpoint", "/parent.pt"]
            )

    def test_schedule_validation_and_manifests_retain_round_contract(self) -> None:
        cli._validate_schedule(6, 3)  # noqa: SLF001
        with self.assertRaisesRegex(ValueError, "critic_epochs must equal"):
            cli._validate_schedule(6, 2)  # noqa: SLF001

        runner = SimpleNamespace(
            critic_epochs=6,
            actor_equivalent_epochs=3,
            POLICY_UPDATE_PERIOD=2,
            ROUND_EPISODES=50,
            MAX_EPISODES=200,
            round_index=2,
            new_episode_count=37,
            batches_per_epoch=11,
            total_critic_updates=66,
            total_actor_updates=33,
        )
        self.assertEqual(
            cli._schedule_manifest(runner),  # noqa: SLF001
            {
                "critic_epochs": 6,
                "actor_equivalent_epochs": 3,
                "policy_update_period": 2,
                "round_episodes": 50,
                "max_new_episodes_per_round": 50,
                "max_episodes": 200,
            },
        )
        self.assertEqual(
            cli._round_manifest(runner),  # noqa: SLF001
            {
                "index": 2,
                "new_episodes": 37,
                "batches_per_epoch": 11,
                "critic_updates": 66,
                "actor_updates": 33,
            },
        )

    def test_model_export_is_strictly_verified_and_idempotent(self) -> None:
        torch.manual_seed(7)
        actor = _FakePolicy().eval()
        expected_actor = copy.deepcopy(actor).eval()
        assets = SimpleNamespace(
            action_mean=torch.tensor([1.0, 2.0]),
            action_std=torch.tensor([3.0, 4.0]),
            normalizer_eps=1.0e-8,
            preprocessor=_FakeProcessor("policy_preprocessor.json"),
            postprocessor=_FakeProcessor("policy_postprocessor.json"),
        )
        verified = SimpleNamespace(
            policy=expected_actor,
            action_mean=assets.action_mean.clone(),
            action_std=assets.action_std.clone(),
            normalizer_eps=assets.normalizer_eps,
        )
        learner = SimpleNamespace(actor=actor, device=torch.device("cpu"))

        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "pretrained_model"
            with mock.patch.object(
                cli,
                "load_act_policy_assets",
                return_value=verified,
            ) as loader:
                cli._export_policy_assets(  # noqa: SLF001
                    destination,
                    learner=learner,
                    source_assets=assets,
                )
                cli._export_policy_assets(  # noqa: SLF001
                    destination,
                    learner=learner,
                    source_assets=assets,
                )

            self.assertTrue((destination / "config.json").is_file())
            self.assertTrue((destination / "model.safetensors").is_file())
            self.assertTrue((destination / "policy_preprocessor.json").is_file())
            self.assertTrue((destination / "policy_postprocessor.json").is_file())
            self.assertEqual(loader.call_count, 2)
            self.assertFalse(any(destination.parent.glob(".pretrained_model.*.tmp")))

    def test_unchanged_training_data_identity_exports_model(self) -> None:
        expected_identity = SimpleNamespace(identity="sha256:unchanged")
        revalidated_identity = SimpleNamespace(identity="sha256:unchanged")
        dataset = object()
        learner = object()
        assets = object()
        action_domain = object()
        dataset_root = Path("/dataset")
        checkpoint_root = Path("/checkpoint")
        destination = Path("/output/pretrained_model")

        with (
            mock.patch.object(
                cli,
                "build_act_td3_training_data_identity",
                return_value=revalidated_identity,
            ) as identity_builder,
            mock.patch.object(cli, "_export_policy_assets") as exporter,
        ):
            cli._publish_policy_assets_for_unchanged_training_data(  # noqa: SLF001
                destination,
                learner=learner,
                source_assets=assets,
                expected_identity=expected_identity,
                dataset=dataset,
                dataset_root=dataset_root,
                act_checkpoint_root=checkpoint_root,
                action_domain=action_domain,
                robot_type="ffw_sg2_rev1",
                video_backend="pyav",
            )

        identity_builder.assert_called_once_with(
            dataset,
            dataset_root=dataset_root,
            act_checkpoint_root=checkpoint_root,
            action_domain=action_domain,
            robot_type="ffw_sg2_rev1",
            video_backend="pyav",
        )
        exporter.assert_called_once_with(
            destination,
            learner=learner,
            source_assets=assets,
        )

    def test_changed_training_data_identity_blocks_model_export(self) -> None:
        expected_identity = SimpleNamespace(identity="sha256:before")
        changed_identity = SimpleNamespace(identity="sha256:after")

        with (
            mock.patch.object(
                cli,
                "build_act_td3_training_data_identity",
                return_value=changed_identity,
            ),
            mock.patch.object(cli, "_export_policy_assets") as exporter,
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "training data identity changed during training",
            ):
                cli._publish_policy_assets_for_unchanged_training_data(  # noqa: SLF001
                    Path("/output/pretrained_model"),
                    learner=object(),
                    source_assets=object(),
                    expected_identity=expected_identity,
                    dataset=object(),
                    dataset_root=Path("/dataset"),
                    act_checkpoint_root=Path("/checkpoint"),
                    action_domain=object(),
                    robot_type="ffw_sg2_rev1",
                    video_backend="pyav",
                )

        exporter.assert_not_called()

    def test_output_directory_is_immutable_and_outside_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset"
            actor = root / "actor"
            robot = root / "config" / "robot.yaml"
            dataset.mkdir()
            actor.mkdir()
            robot.parent.mkdir()
            robot.write_text("robot: test\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "outside dataset"):
                cli._output_directory(  # noqa: SLF001
                    dataset / "output",
                    resume=False,
                    inputs=(dataset, actor, robot),
                )

            output = root / "round"
            output.mkdir()
            with self.assertRaisesRegex(FileExistsError, "already exists"):
                cli._output_directory(  # noqa: SLF001
                    output,
                    resume=False,
                    inputs=(dataset, actor, robot),
                )
            with self.assertRaisesRegex(FileNotFoundError, "resume checkpoint"):
                cli._output_directory(  # noqa: SLF001
                    output,
                    resume=True,
                    inputs=(dataset, actor, robot),
                )


if __name__ == "__main__":
    unittest.main()
