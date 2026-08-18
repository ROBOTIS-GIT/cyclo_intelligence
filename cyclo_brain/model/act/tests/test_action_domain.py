"""Tests for the validated ACT physical action-domain adapter."""

from __future__ import annotations

import json
import tempfile
import textwrap
import unittest
from pathlib import Path

import torch
import yaml
from torch import nn

from cyclo_brain.model.act.action_domain import (
    ACTPhysicalActionDomain,
    build_act_execution_projector,
    load_act_physical_action_domain,
)
from cyclo_brain.model.act.assets import ACTPolicyAssets


class _SyntheticDomainFiles:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.info = root / "info.json"
        self.config = root / "robot.yaml"
        self.urdf = root / "robot.urdf"

    def write(
        self,
        *,
        dataset_names: tuple[str, ...] = ("joint_a", "joint_b", "joint_c"),
        first_group: tuple[str, ...] = ("joint_a", "joint_b"),
        second_group: tuple[str, ...] = ("joint_c",),
        first_msg_type: str | None = "trajectory_msgs/msg/JointTrajectory",
        second_msg_type: str | None = "trajectory_msgs/msg/JointTrajectory",
        mobile_msg_type: str | None = "geometry_msgs/msg/Twist",
        urdf_body: str | None = None,
    ) -> None:
        self.info.write_text(
            json.dumps(
                {
                    "features": {
                        "action": {
                            "shape": [len(dataset_names)],
                            "names": list(dataset_names),
                            "min": [99.0] * len(dataset_names),
                            "max": [100.0] * len(dataset_names),
                        }
                    },
                    # These values must never become execution limits.
                    "stats": {"action": {"min": [99.0], "max": [100.0]}},
                }
            ),
            encoding="utf-8",
        )
        self.config.write_text(
            yaml.safe_dump(
                {
                    "orchestrator": {
                        "ros__parameters": {
                            "test_robot": {
                                "urdf_path": "robot.urdf",
                                "action": {
                                    "arm": {
                                        "topic": "/arm",
                                        "msg_type": first_msg_type,
                                        "joint_names": list(first_group),
                                    },
                                    "head": {
                                        "topic": "/head",
                                        "msg_type": second_msg_type,
                                        "joint_names": list(second_group),
                                    },
                                    "mobile": {
                                        "topic": "/cmd_vel",
                                        "msg_type": mobile_msg_type,
                                        "joint_names": [
                                            "linear_x",
                                            "linear_y",
                                            "angular_z",
                                        ],
                                    },
                                },
                            }
                        }
                    }
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        self.urdf.write_text(
            urdf_body
            or textwrap.dedent(
                """
                <robot name="test">
                  <joint name="joint_c" type="prismatic">
                    <limit lower="-0.5" upper="0.0" effort="1" velocity="1"/>
                  </joint>
                  <joint name="joint_a" type="revolute">
                    <limit lower="-1.0" upper="2.0" effort="1" velocity="1"/>
                  </joint>
                  <joint name="joint_b" type="revolute">
                    <limit lower="0.0" upper="3.0" effort="1" velocity="1"/>
                  </joint>
                </robot>
                """
            ),
            encoding="utf-8",
        )


class ACTPhysicalActionDomainTest(unittest.TestCase):
    def test_loads_complete_leading_groups_and_urdf_limits(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            files = _SyntheticDomainFiles(Path(directory))
            files.write()

            domain = load_act_physical_action_domain(
                files.info,
                files.config,
                robot_type="test_robot",
            )

            self.assertEqual(domain.names, ("joint_a", "joint_b", "joint_c"))
            self.assertEqual(domain.action_groups, ("arm", "head"))
            self.assertFalse(bool(domain.passthrough_mask.any()))
            self.assertEqual(domain.urdf_path, files.urdf.resolve())
            torch.testing.assert_close(
                domain.physical_low,
                torch.tensor([-1.0, 0.0, -0.5], dtype=torch.float64),
            )
            torch.testing.assert_close(
                domain.physical_high,
                torch.tensor([2.0, 3.0, 0.0], dtype=torch.float64),
            )

            mutated = domain.physical_low
            mutated[0] = 999.0
            self.assertEqual(float(domain.physical_low[0]), -1.0)

    def test_rejects_reordering_and_partial_action_groups(self) -> None:
        cases = (
            (("joint_b", "joint_a", "joint_c"), "ordered prefix"),
            (("joint_a",), "group boundary"),
            (("joint_a", "joint_b", "linear_x"), "ordered prefix"),
        )
        for dataset_names, message in cases:
            with self.subTest(dataset_names=dataset_names):
                with tempfile.TemporaryDirectory() as directory:
                    files = _SyntheticDomainFiles(Path(directory))
                    files.write(dataset_names=dataset_names)
                    with self.assertRaisesRegex(ValueError, message):
                        load_act_physical_action_domain(files.info, files.config)

    def test_rejects_duplicate_missing_and_unbounded_joints(self) -> None:
        bad_urdfs = {
            "duplicate": """
                <robot name="test">
                  <joint name="joint_a" type="revolute"><limit lower="-1" upper="1"/></joint>
                  <joint name="joint_a" type="revolute"><limit lower="-1" upper="1"/></joint>
                  <joint name="joint_b" type="revolute"><limit lower="-1" upper="1"/></joint>
                  <joint name="joint_c" type="revolute"><limit lower="-1" upper="1"/></joint>
                </robot>
            """,
            "missing": """
                <robot name="test">
                  <joint name="joint_a" type="revolute"><limit lower="-1" upper="1"/></joint>
                  <joint name="joint_b" type="revolute"><limit lower="-1" upper="1"/></joint>
                </robot>
            """,
            "continuous": """
                <robot name="test">
                  <joint name="joint_a" type="continuous"/>
                  <joint name="joint_b" type="revolute"><limit lower="-1" upper="1"/></joint>
                  <joint name="joint_c" type="revolute"><limit lower="-1" upper="1"/></joint>
                </robot>
            """,
            "finite": """
                <robot name="test">
                  <joint name="joint_a" type="revolute"><limit lower="-inf" upper="1"/></joint>
                  <joint name="joint_b" type="revolute"><limit lower="-1" upper="1"/></joint>
                  <joint name="joint_c" type="revolute"><limit lower="-1" upper="1"/></joint>
                </robot>
            """,
        }
        for expected, urdf in bad_urdfs.items():
            with self.subTest(expected=expected):
                with tempfile.TemporaryDirectory() as directory:
                    files = _SyntheticDomainFiles(Path(directory))
                    files.write(urdf_body=textwrap.dedent(urdf))
                    with self.assertRaisesRegex(ValueError, expected):
                        load_act_physical_action_domain(files.info, files.config)

    def test_loads_complete_twist_group_as_unbounded_passthrough(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            files = _SyntheticDomainFiles(Path(directory))
            files.write(
                dataset_names=(
                    "joint_a",
                    "joint_b",
                    "joint_c",
                    "linear_x",
                    "linear_y",
                    "angular_z",
                )
            )

            domain = load_act_physical_action_domain(files.info, files.config)

            self.assertEqual(domain.action_groups, ("arm", "head", "mobile"))
            torch.testing.assert_close(
                domain.passthrough_mask,
                torch.tensor([False, False, False, True, True, True]),
            )
            torch.testing.assert_close(
                domain.physical_low,
                torch.tensor([-1.0, 0.0, -0.5, 0.0, 0.0, 0.0], dtype=torch.float64),
            )
            torch.testing.assert_close(
                domain.physical_high,
                torch.tensor([2.0, 3.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float64),
            )

    def test_rejects_unknown_action_message_type(self) -> None:
        for msg_type in (None, "std_msgs/msg/Float64MultiArray"):
            with self.subTest(msg_type=msg_type):
                with tempfile.TemporaryDirectory() as directory:
                    files = _SyntheticDomainFiles(Path(directory))
                    files.write(first_msg_type=msg_type)
                    with self.assertRaisesRegex(ValueError, "unsupported msg_type"):
                        load_act_physical_action_domain(files.info, files.config)

    def test_rejects_nonzero_passthrough_bound_placeholders(self) -> None:
        with self.assertRaisesRegex(ValueError, "zero placeholders"):
            ACTPhysicalActionDomain(
                names=("joint", "linear_x"),
                action_groups=("arm", "mobile"),
                dataset_info_path=Path("info.json"),
                robot_config_path=Path("robot.yaml"),
                urdf_path=Path("robot.urdf"),
                _physical_low=torch.tensor([-1.0, -1.0]),
                _physical_high=torch.tensor([1.0, 1.0]),
                _passthrough_mask=torch.tensor([False, True]),
            )

    def test_builds_continuous_projector_on_policy_dtype_and_device(self) -> None:
        domain = ACTPhysicalActionDomain(
            names=("a", "b"),
            action_groups=("arm",),
            dataset_info_path=Path("info.json"),
            robot_config_path=Path("robot.yaml"),
            urdf_path=Path("robot.urdf"),
            _physical_low=torch.tensor([-1.0, 0.0]),
            _physical_high=torch.tensor([1.0, 2.0]),
        )
        policy = nn.Linear(2, 2, dtype=torch.float64)
        assets = ACTPolicyAssets(
            policy=policy,
            preprocessor=object(),
            postprocessor=object(),
            _action_mean=torch.tensor([0.25, 0.5]),
            _action_std=torch.tensor([0.5, 0.25]),
            normalizer_eps=1.0e-8,
        )

        projector = build_act_execution_projector(assets, domain)

        self.assertEqual(projector.action_mean.dtype, torch.float64)
        self.assertEqual(projector.action_mean.device, next(policy.parameters()).device)
        self.assertFalse(bool(projector.binary_mask.any()))
        self.assertFalse(bool(projector.passthrough_mask.any()))
        torch.testing.assert_close(
            projector.physical_low,
            torch.tensor([-1.0, 0.0], dtype=torch.float64),
        )

    def test_build_projector_forwards_passthrough_mask(self) -> None:
        domain = ACTPhysicalActionDomain(
            names=("joint", "linear_x"),
            action_groups=("arm", "mobile"),
            dataset_info_path=Path("info.json"),
            robot_config_path=Path("robot.yaml"),
            urdf_path=Path("robot.urdf"),
            _physical_low=torch.tensor([-1.0, 0.0]),
            _physical_high=torch.tensor([1.0, 0.0]),
            _passthrough_mask=torch.tensor([False, True]),
        )
        assets = ACTPolicyAssets(
            policy=nn.Linear(2, 2),
            preprocessor=object(),
            postprocessor=object(),
            _action_mean=torch.zeros(2),
            _action_std=torch.ones(2),
            normalizer_eps=1.0e-8,
        )

        projector = build_act_execution_projector(assets, domain)

        torch.testing.assert_close(
            projector.passthrough_mask,
            torch.tensor([False, True]),
        )

    def test_projector_rejects_checkpoint_domain_dimension_mismatch(self) -> None:
        domain = ACTPhysicalActionDomain(
            names=("a",),
            action_groups=("arm",),
            dataset_info_path=Path("info.json"),
            robot_config_path=Path("robot.yaml"),
            urdf_path=Path("robot.urdf"),
            _physical_low=torch.tensor([-1.0]),
            _physical_high=torch.tensor([1.0]),
        )
        assets = ACTPolicyAssets(
            policy=nn.Linear(2, 2),
            preprocessor=object(),
            postprocessor=object(),
            _action_mean=torch.zeros(2),
            _action_std=torch.ones(2),
            normalizer_eps=1.0e-8,
        )
        with self.assertRaisesRegex(ValueError, "dimensions disagree"):
            build_act_execution_projector(assets, domain)


_REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
_REAL_DATASET_INFO = (
    _REPOSITORY_ROOT
    / "docker/workspace/dataset"
    / "Task_900011_900012_stage3_x_900006_900007_merged_no_mobile"
    / "meta/info.json"
)
_REAL_ROBOT_CONFIG = (
    _REPOSITORY_ROOT / "shared/shared/robot_configs/ffw_sg2_rev1_config.yaml"
)
_REAL_22D_DATASET_INFO = (
    _REPOSITORY_ROOT
    / "docker/workspace/lerobot"
    / "Task_20260814_090416_inference_MCAP_lerobot_v30_actionfix_atomic"
    / "meta/info.json"
)


@unittest.skipUnless(
    _REAL_DATASET_INFO.is_file() and _REAL_ROBOT_CONFIG.is_file(),
    "local FFW SG2 ACT dataset or robot config is unavailable",
)
class FFWSG2ActualActionDomainSmokeTest(unittest.TestCase):
    def test_actual_19d_checkpoint_dataset_contract(self) -> None:
        domain = load_act_physical_action_domain(
            _REAL_DATASET_INFO,
            _REAL_ROBOT_CONFIG,
            robot_type="ffw_sg2_rev1",
        )

        self.assertEqual(domain.action_dim, 19)
        self.assertEqual(
            domain.action_groups,
            ("arm_left", "arm_right", "head", "lift"),
        )
        self.assertEqual(domain.names[7], "gripper_l_joint1")
        self.assertEqual(domain.names[15], "gripper_r_joint1")
        self.assertEqual(domain.names[16:], ("head_joint1", "head_joint2", "lift_joint"))
        torch.testing.assert_close(
            domain.physical_low,
            torch.tensor(
                [
                    -3.14,
                    0.0,
                    -3.14,
                    -2.9361,
                    -3.14,
                    -1.57,
                    -1.8201,
                    0.0,
                    -3.14,
                    -3.14,
                    -3.14,
                    -2.9361,
                    -3.14,
                    -1.57,
                    -1.5804,
                    0.0,
                    -0.2317,
                    -0.35,
                    -0.5,
                ],
                dtype=torch.float64,
            ),
        )
        torch.testing.assert_close(
            domain.physical_high,
            torch.tensor(
                [
                    3.14,
                    3.14,
                    3.14,
                    1.0786,
                    3.14,
                    1.57,
                    1.5804,
                    1.1,
                    3.14,
                    0.0,
                    3.14,
                    1.0786,
                    3.14,
                    1.57,
                    1.8201,
                    1.1,
                    0.6951,
                    0.35,
                    0.0,
                ],
                dtype=torch.float64,
            ),
        )

    @unittest.skipUnless(
        _REAL_22D_DATASET_INFO.is_file(),
        "local 22-D FFW SG2 inference dataset is unavailable",
    )
    def test_actual_22d_dataset_marks_only_mobile_as_passthrough(self) -> None:
        domain = load_act_physical_action_domain(
            _REAL_22D_DATASET_INFO,
            _REAL_ROBOT_CONFIG,
            robot_type="ffw_sg2_rev1",
        )

        self.assertEqual(domain.action_dim, 22)
        self.assertEqual(
            domain.action_groups,
            ("arm_left", "arm_right", "head", "lift", "mobile"),
        )
        torch.testing.assert_close(
            domain.passthrough_mask,
            torch.tensor([False] * 19 + [True] * 3),
        )
        torch.testing.assert_close(
            domain.physical_low[-3:],
            torch.zeros(3, dtype=torch.float64),
        )
        torch.testing.assert_close(
            domain.physical_high[-3:],
            torch.zeros(3, dtype=torch.float64),
        )


if __name__ == "__main__":
    unittest.main()
