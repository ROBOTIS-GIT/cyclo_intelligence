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

from cyclo_brain.algorithm.rl.act_td3 import ACTTD3Config
from cyclo_brain.algorithm.rl.act_td3 import RLMetricHistoryPoint
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
        self.assertEqual(args.dataset_root, [Path("/dataset")])
        self.assertFalse(args.resume)
        self.assertIsNone(args.parent_checkpoint)
        self.assertIsNone(args.max_round_critic_updates)
        self.assertEqual(args.critic_epochs, 10)
        self.assertEqual(args.actor_equivalent_epochs, 5)
        self.assertFalse(args.allow_partial_round)
        self.assertIsNone(args.actor_trainable_groups)
        self.assertEqual(args.actor_objective, "td3_bc")

        multiple = parser.parse_args(
            [*required, "--dataset-root", "/dataset_epoch_0001"]
        )
        self.assertEqual(
            multiple.dataset_root,
            [Path("/dataset"), Path("/dataset_epoch_0001")],
        )
        self.assertEqual(
            cli._dataset_root_arguments(Path("/legacy_single")),  # noqa: SLF001
            (Path("/legacy_single"),),
        )

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

        selected = parser.parse_args(
            [
                *required,
                "--actor-trainable-group",
                "action_decoder",
                "--actor-trainable-group",
                "visual_backbone",
            ]
        )
        self.assertEqual(
            selected.actor_trainable_groups,
            ["action_decoder", "visual_backbone"],
        )
        self.assertEqual(
            ACTTD3Config(
                actor_trainable_groups=tuple(selected.actor_trainable_groups)
            ).actor_trainable_groups,
            ("visual_backbone", "action_decoder"),
        )

        pure_td3 = parser.parse_args([*required, "--actor-objective", "td3"])
        self.assertEqual(pure_td3.actor_objective, "td3")
        with self.assertRaises(SystemExit):
            parser.parse_args([*required, "--actor-objective", "sac"])

        with self.assertRaises(SystemExit):
            parser.parse_args(
                [*required, "--resume", "--parent-checkpoint", "/parent.pt"]
            )

    def test_schedule_validation_and_manifests_retain_round_contract(self) -> None:
        self.assertEqual(cli._validate_schedule(6, 3), 2)  # noqa: SLF001
        self.assertEqual(cli._validate_schedule(1, 1), 1)  # noqa: SLF001
        with self.assertRaisesRegex(ValueError, "exact integer multiple"):
            cli._validate_schedule(5, 3)  # noqa: SLF001

        runner = SimpleNamespace(
            critic_epochs=6,
            actor_equivalent_epochs=3,
            policy_update_period=2,
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

        progress = cli.ACTTD3OfflineTrainingProgress(
            status="stopped",
            round_index=2,
            episode_count=87,
            completed_epochs=1,
            total_epochs=6,
            completed_critic_updates=11,
            total_critic_updates=66,
            completed_actor_updates=5,
            total_actor_updates=33,
            percentage=16.7,
            critic_loss=0.1,
            actor_loss=0.2,
            elapsed_seconds=12.0,
            eta_seconds=None,
            durable_critic_updates=11,
            checkpoint_path="/output/training_state/act_td3.pt",
            rl_metric_history=(
                RLMetricHistoryPoint(
                    rl_epoch=1,
                    actor_loss_mean=-0.2,
                    critic_loss_mean=0.3,
                    replay_average_reward=0.75,
                ),
                RLMetricHistoryPoint(
                    rl_epoch=2,
                    actor_loss_mean=0.2,
                    critic_loss_mean=0.1,
                    replay_average_reward=0.8,
                ),
            ),
        )
        identity = SimpleNamespace(
            identity="sha256:identity",
            file_count=3,
            byte_count=42,
            component_sha256={"data": "abc"},
            virtual_contract={"data_roots": ["/dataset"]},
        )
        result_manifest = cli._result_manifest(  # noqa: SLF001
            progress,
            actor_trainable_groups=("visual_backbone", "action_decoder"),
            runner=runner,
            identity=identity,
            model_directory=None,
            batch_size=8,
            critic_source="parent_checkpoint",
            critic_checkpoint=Path("/output/parent.pt"),
        )
        self.assertEqual(result_manifest["batch_size"], 8)
        self.assertEqual(result_manifest["algorithm"], "td3")
        self.assertEqual(result_manifest["actor_objective"], "td3_bc")
        self.assertEqual(result_manifest["status"], "stopped")
        self.assertIsNone(result_manifest["model_path"])
        self.assertEqual(result_manifest["critic_source"], "parent_checkpoint")
        self.assertEqual(
            result_manifest["critic_checkpoint"],
            "/output/parent.pt",
        )
        self.assertEqual(
            result_manifest["rl_metric_history"],
            (
                {
                    "rl_epoch": 1,
                    "actor_loss_mean": -0.2,
                    "critic_loss_mean": 0.3,
                    "replay_average_reward": 0.75,
                },
                {
                    "rl_epoch": 2,
                    "actor_loss_mean": 0.2,
                    "critic_loss_mean": 0.1,
                    "replay_average_reward": 0.8,
                },
            ),
        )
        with mock.patch.object(cli, "_json_line") as emit:
            cli._progress_line(progress)  # noqa: SLF001
        emitted = emit.call_args.args[0]
        self.assertEqual(emitted["event"], "progress")
        self.assertEqual(
            emitted["rl_metric_history"],
            result_manifest["rl_metric_history"],
        )

    def test_full_td3_continuation_wins_without_inspecting_policy_warmup(self) -> None:
        parent = Path("/output/parent.pt")
        with mock.patch.object(
            cli,
            "load_policy_local_warmup_critic",
        ) as warm_loader:
            source, checkpoint = cli._initialize_critic_source(  # noqa: SLF001
                learner=object(),
                replay=object(),
                identity=object(),
                act_checkpoint=Path("/actor"),
                resume_from=parent,
                continuation_source="parent_checkpoint",
            )
        self.assertEqual(source, "parent_checkpoint")
        self.assertEqual(checkpoint, parent)
        warm_loader.assert_not_called()

    def test_policy_warmup_then_random_critic_source_priority(self) -> None:
        latest = Path("/actor/critic/latest.pt")
        with mock.patch.object(
            cli,
            "load_policy_local_warmup_critic",
            side_effect=(latest, None),
        ) as warm_loader:
            warm = cli._initialize_critic_source(  # noqa: SLF001
                learner=object(),
                replay=object(),
                identity=object(),
                act_checkpoint=Path("/actor"),
                resume_from=None,
                continuation_source=None,
            )
            random = cli._initialize_critic_source(  # noqa: SLF001
                learner=object(),
                replay=object(),
                identity=object(),
                act_checkpoint=Path("/actor"),
                resume_from=None,
                continuation_source=None,
            )
        self.assertEqual(warm, ("policy_warmup", latest))
        self.assertEqual(random, ("random", None))
        self.assertEqual(warm_loader.call_count, 2)

    def test_same_round_resume_has_distinct_critic_source(self) -> None:
        checkpoint_path = Path("/output/training_state/act_td3.pt")
        with mock.patch.object(
            cli,
            "load_policy_local_warmup_critic",
        ) as warm_loader:
            source = cli._initialize_critic_source(  # noqa: SLF001
                learner=object(),
                replay=object(),
                identity=object(),
                act_checkpoint=Path("/actor"),
                resume_from=checkpoint_path,
                continuation_source="resume_checkpoint",
            )
        self.assertEqual(source, ("resume_checkpoint", checkpoint_path))
        warm_loader.assert_not_called()

    def test_continuation_source_requires_a_matching_checkpoint(self) -> None:
        with self.assertRaisesRegex(ValueError, "has no checkpoint"):
            cli._initialize_critic_source(  # noqa: SLF001
                learner=object(),
                replay=object(),
                identity=object(),
                act_checkpoint=Path("/actor"),
                resume_from=None,
                continuation_source="parent_checkpoint",
            )
        with self.assertRaisesRegex(ValueError, "source is invalid"):
            cli._initialize_critic_source(  # noqa: SLF001
                learner=object(),
                replay=object(),
                identity=object(),
                act_checkpoint=Path("/actor"),
                resume_from=Path("/output/state.pt"),
                continuation_source="policy_warmup",
            )

    def test_main_turns_sigint_into_cooperative_stop_request(self) -> None:
        installed: dict[int, object] = {}
        signal_calls: list[tuple[int, object]] = []
        previous = {
            cli.signal.SIGINT: object(),
            cli.signal.SIGTERM: object(),
        }

        def fake_signal(signum: int, handler: object) -> None:
            installed[signum] = handler
            signal_calls.append((signum, handler))

        def fake_run_from_args(_args: object, *, should_stop) -> object:
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
            mock.patch.object(
                cli,
                "run_from_args",
                side_effect=fake_run_from_args,
            ),
        ):
            self.assertEqual(cli.main([]), 0)

        self.assertEqual(signal_calls[-2:], [
            (cli.signal.SIGINT, previous[cli.signal.SIGINT]),
            (cli.signal.SIGTERM, previous[cli.signal.SIGTERM]),
        ])

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
