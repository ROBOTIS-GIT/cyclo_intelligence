"""Validated physical action domains for ACT execution and ACT-TD3."""

from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import Tensor

from .action_projection import ACTExecutionProjector
from .assets import ACTPolicyAssets


@dataclass(frozen=True)
class ACTPhysicalActionDomain:
    """Auditable ACT action order and execution-domain semantics.

    Tensor fields are kept private and exposed through detached clones so a
    caller cannot mutate the validated contract in place. Physical bounds are
    meaningful only where ``passthrough_mask`` is false; passthrough command
    dimensions use validated zero placeholders in both bound vectors.
    """

    names: tuple[str, ...]
    action_groups: tuple[str, ...]
    dataset_info_path: Path
    robot_config_path: Path
    urdf_path: Path
    _physical_low: Tensor = field(repr=False)
    _physical_high: Tensor = field(repr=False)
    _passthrough_mask: Tensor | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.names, Sequence) or isinstance(self.names, (str, bytes)):
            raise TypeError("ACT physical action names must be a sequence")
        if not isinstance(self.action_groups, Sequence) or isinstance(
            self.action_groups, (str, bytes)
        ):
            raise TypeError("ACT physical action groups must be a sequence")
        names = tuple(self.names)
        groups = tuple(self.action_groups)
        if not names or any(not isinstance(name, str) or not name for name in names):
            raise ValueError("ACT physical action names must be non-empty strings")
        if len(set(names)) != len(names):
            raise ValueError("ACT physical action names must be unique")
        if not groups or any(not isinstance(name, str) or not name for name in groups):
            raise ValueError("ACT physical action group names must be non-empty strings")
        if len(set(groups)) != len(groups):
            raise ValueError("ACT physical action group names must be unique")

        low = (
            torch.as_tensor(self._physical_low, dtype=torch.float64)
            .detach()
            .clone()
            .reshape(-1)
        )
        high = (
            torch.as_tensor(self._physical_high, dtype=torch.float64)
            .detach()
            .clone()
            .reshape(-1)
        )
        expected_shape = (len(names),)
        if low.shape != expected_shape or high.shape != expected_shape:
            raise ValueError(
                "ACT physical bounds must match the validated action-name dimension"
            )
        if self._passthrough_mask is None:
            passthrough_mask = torch.zeros(expected_shape, dtype=torch.bool)
        else:
            raw_passthrough_mask = torch.as_tensor(self._passthrough_mask).detach()
            if raw_passthrough_mask.dtype != torch.bool:
                raise TypeError("ACT passthrough_mask must be boolean")
            passthrough_mask = raw_passthrough_mask.clone().reshape(-1)
            if passthrough_mask.shape != expected_shape:
                raise ValueError(
                    "ACT passthrough_mask must match the validated action-name dimension"
                )
        if not bool(torch.isfinite(low).all()) or not bool(torch.isfinite(high).all()):
            raise ValueError("ACT physical bounds must be finite")
        bounded_mask = ~passthrough_mask
        if bool((low[bounded_mask] >= high[bounded_mask]).any()):
            raise ValueError("ACT physical lower bounds must be strictly below upper bounds")
        if bool((low[passthrough_mask] != 0.0).any()) or bool(
            (high[passthrough_mask] != 0.0).any()
        ):
            raise ValueError(
                "ACT passthrough physical bounds must use explicit zero placeholders"
            )

        object.__setattr__(self, "names", names)
        object.__setattr__(self, "action_groups", groups)
        object.__setattr__(self, "dataset_info_path", Path(self.dataset_info_path))
        object.__setattr__(self, "robot_config_path", Path(self.robot_config_path))
        object.__setattr__(self, "urdf_path", Path(self.urdf_path))
        object.__setattr__(self, "_physical_low", low)
        object.__setattr__(self, "_physical_high", high)
        object.__setattr__(self, "_passthrough_mask", passthrough_mask)

    @property
    def action_dim(self) -> int:
        return len(self.names)

    @property
    def physical_low(self) -> Tensor:
        return self._physical_low.detach().clone()

    @property
    def physical_high(self) -> Tensor:
        return self._physical_high.detach().clone()

    @property
    def passthrough_mask(self) -> Tensor:
        assert self._passthrough_mask is not None
        return self._passthrough_mask.detach().clone()


def _read_mapping(path: Path, *, source: str, loader: Any) -> Mapping[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"ACT {source} file does not exist: {path}")
    try:
        with path.open("r", encoding="utf-8") as stream:
            value = loader(stream)
    except (json.JSONDecodeError, yaml.YAMLError) as error:
        raise ValueError(f"ACT {source} file is invalid: {path}") from error
    if not isinstance(value, Mapping):
        raise ValueError(f"ACT {source} root must be a mapping")
    return value


def _dataset_action_names(info: Mapping[str, Any]) -> tuple[str, ...]:
    features = info.get("features")
    action = features.get("action") if isinstance(features, Mapping) else None
    if not isinstance(action, Mapping):
        raise ValueError("ACT dataset info must define features.action")

    shape = action.get("shape")
    if (
        not isinstance(shape, Sequence)
        or isinstance(shape, (str, bytes))
        or len(shape) != 1
        or isinstance(shape[0], bool)
        or not isinstance(shape[0], int)
        or shape[0] < 1
    ):
        raise ValueError("ACT dataset features.action.shape must be [action_dim]")

    raw_names = action.get("names")
    if (
        not isinstance(raw_names, Sequence)
        or isinstance(raw_names, (str, bytes))
        or len(raw_names) != shape[0]
    ):
        raise ValueError(
            "ACT dataset features.action.names must contain exactly action_dim names"
        )
    names = tuple(raw_names)
    if any(not isinstance(name, str) or not name for name in names):
        raise ValueError("ACT dataset action names must be non-empty strings")
    if len(set(names)) != len(names):
        raise ValueError("ACT dataset action names must be unique")
    return names


def _robot_section(
    document: Mapping[str, Any],
    *,
    robot_type: str | None,
) -> Mapping[str, Any]:
    orchestrator = document.get("orchestrator")
    parameters = (
        orchestrator.get("ros__parameters")
        if isinstance(orchestrator, Mapping)
        else None
    )
    if not isinstance(parameters, Mapping):
        raise ValueError(
            "ACT robot config must define orchestrator.ros__parameters"
        )

    if robot_type is not None:
        if not isinstance(robot_type, str) or not robot_type:
            raise ValueError("ACT robot_type must be a non-empty string")
        section = parameters.get(robot_type)
        if not isinstance(section, Mapping):
            raise ValueError(f"ACT robot config has no section for {robot_type!r}")
        return section

    candidates = [
        value
        for value in parameters.values()
        if isinstance(value, Mapping)
        and isinstance(value.get("action"), Mapping)
        and isinstance(value.get("urdf_path"), str)
    ]
    if len(candidates) != 1:
        raise ValueError(
            "ACT robot config must contain exactly one action/URDF robot section "
            "when robot_type is omitted"
        )
    return candidates[0]


def _matched_action_groups(
    section: Mapping[str, Any],
    dataset_names: tuple[str, ...],
) -> tuple[tuple[str, ...], Tensor]:
    actions = section.get("action")
    if not isinstance(actions, Mapping) or not actions:
        raise ValueError("ACT robot config must define non-empty action groups")

    flattened: list[str] = []
    passthrough: list[bool] = []
    boundaries: dict[int, tuple[tuple[str, ...], tuple[bool, ...]]] = {}
    group_names: list[str] = []
    for group_name, group in actions.items():
        if not isinstance(group_name, str) or not group_name:
            raise ValueError("ACT robot action group names must be non-empty strings")
        if not isinstance(group, Mapping):
            raise ValueError(f"ACT robot action group {group_name!r} must be a mapping")
        msg_type = group.get("msg_type")
        if msg_type not in {
            "trajectory_msgs/msg/JointTrajectory",
            "geometry_msgs/msg/Twist",
        }:
            raise ValueError(
                f"ACT robot action group {group_name!r} has unsupported msg_type "
                f"{msg_type!r}"
            )
        joint_names = group.get("joint_names")
        if (
            not isinstance(joint_names, Sequence)
            or isinstance(joint_names, (str, bytes))
            or not joint_names
            or any(not isinstance(name, str) or not name for name in joint_names)
        ):
            raise ValueError(
                f"ACT robot action group {group_name!r} must define joint_names"
            )
        group_names.append(group_name)
        flattened.extend(joint_names)
        passthrough.extend(
            [msg_type == "geometry_msgs/msg/Twist"] * len(joint_names)
        )
        boundaries[len(flattened)] = (tuple(group_names), tuple(passthrough))

    if len(set(flattened)) != len(flattened):
        raise ValueError("ACT robot action names must be unique across groups")

    dataset_width = len(dataset_names)
    if tuple(flattened[:dataset_width]) != dataset_names:
        mismatch = next(
            (
                index
                for index, (dataset_name, robot_name) in enumerate(
                    zip(dataset_names, flattened, strict=False)
                )
                if dataset_name != robot_name
            ),
            min(dataset_width, len(flattened)),
        )
        raise ValueError(
            "ACT dataset action order must match the ordered prefix of robot "
            f"action groups; first mismatch is index {mismatch}"
        )
    if dataset_width not in boundaries:
        raise ValueError(
            "ACT dataset action prefix must end at a complete robot action-group boundary"
        )
    matched_groups, matched_passthrough = boundaries[dataset_width]
    return matched_groups, torch.tensor(matched_passthrough, dtype=torch.bool)


def _resolve_urdf_path(section: Mapping[str, Any], robot_config_path: Path) -> Path:
    raw_path = section.get("urdf_path")
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError("ACT robot config must define a non-empty urdf_path")
    urdf_path = Path(raw_path)
    if not urdf_path.is_absolute():
        urdf_path = robot_config_path.parent / urdf_path
    urdf_path = urdf_path.resolve()
    if not urdf_path.is_file():
        raise FileNotFoundError(f"ACT robot URDF does not exist: {urdf_path}")
    return urdf_path


def _tag_name(element: ET.Element) -> str:
    return element.tag.rsplit("}", 1)[-1]


def _urdf_hard_limits(
    urdf_path: Path,
    action_names: tuple[str, ...],
) -> tuple[Tensor, Tensor]:
    try:
        root = ET.parse(urdf_path).getroot()
    except (ET.ParseError, OSError) as error:
        raise ValueError(f"ACT robot URDF is invalid: {urdf_path}") from error

    requested = set(action_names)
    joints: dict[str, list[ET.Element]] = {name: [] for name in action_names}
    for element in root.iter():
        if _tag_name(element) != "joint":
            continue
        name = element.get("name")
        if name in requested:
            joints[name].append(element)

    duplicates = [name for name, values in joints.items() if len(values) > 1]
    if duplicates:
        raise ValueError(f"ACT robot URDF contains duplicate action joints: {duplicates}")
    missing = [name for name, values in joints.items() if not values]
    if missing:
        raise ValueError(f"ACT robot URDF is missing action joints: {missing}")

    lower_values: list[float] = []
    upper_values: list[float] = []
    for name in action_names:
        joint = joints[name][0]
        joint_type = joint.get("type", "")
        if joint_type == "continuous":
            raise ValueError(
                f"ACT action joint {name!r} is continuous and has no finite hard limits"
            )
        if joint_type not in {"revolute", "prismatic"}:
            raise ValueError(
                f"ACT action joint {name!r} must be revolute or prismatic, got {joint_type!r}"
            )
        limits = [child for child in joint if _tag_name(child) == "limit"]
        if len(limits) != 1:
            raise ValueError(
                f"ACT action joint {name!r} must define exactly one URDF limit"
            )
        try:
            lower = float(limits[0].attrib["lower"])
            upper = float(limits[0].attrib["upper"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"ACT action joint {name!r} must define numeric lower/upper limits"
            ) from error
        if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
            raise ValueError(
                f"ACT action joint {name!r} must have finite lower < upper limits"
            )
        lower_values.append(lower)
        upper_values.append(upper)

    return (
        torch.tensor(lower_values, dtype=torch.float64),
        torch.tensor(upper_values, dtype=torch.float64),
    )


def load_act_physical_action_domain(
    dataset_info_path: str | Path,
    robot_config_path: str | Path,
    *,
    robot_type: str | None = None,
) -> ACTPhysicalActionDomain:
    """Load a named ACT domain from dataset metadata, robot YAML, and URDF.

    A dataset may omit only complete trailing robot action groups. Its names
    must exactly equal the concatenation of one or more leading YAML groups.
    This admits the real 19-D SG2 dataset (which omits trailing ``mobile``)
    without guessing arbitrary subsets or reordering any dimension.

    ``trajectory_msgs/msg/JointTrajectory`` dimensions require exact finite
    limits from the URDF selected by the robot configuration.
    ``geometry_msgs/msg/Twist`` dimensions are commands rather than URDF
    joints, so complete Twist groups are marked as passthrough. Their zero
    entries in the physical-bound tensors are representation placeholders,
    not inferred limits. Dataset statistics are intentionally not read.
    """

    info_path = Path(dataset_info_path).expanduser().resolve()
    config_path = Path(robot_config_path).expanduser().resolve()
    info = _read_mapping(info_path, source="dataset info", loader=json.load)
    document = _read_mapping(
        config_path,
        source="robot config",
        loader=yaml.safe_load,
    )
    action_names = _dataset_action_names(info)
    section = _robot_section(document, robot_type=robot_type)
    matched_groups, passthrough_mask = _matched_action_groups(section, action_names)
    urdf_path = _resolve_urdf_path(section, config_path)
    bounded_names = tuple(
        name
        for name, is_passthrough in zip(
            action_names,
            passthrough_mask.tolist(),
            strict=True,
        )
        if not is_passthrough
    )
    bounded_low, bounded_high = _urdf_hard_limits(urdf_path, bounded_names)
    physical_low = torch.zeros(len(action_names), dtype=torch.float64)
    physical_high = torch.zeros(len(action_names), dtype=torch.float64)
    physical_low[~passthrough_mask] = bounded_low
    physical_high[~passthrough_mask] = bounded_high
    return ACTPhysicalActionDomain(
        names=action_names,
        action_groups=matched_groups,
        dataset_info_path=info_path,
        robot_config_path=config_path,
        urdf_path=urdf_path,
        _physical_low=physical_low,
        _physical_high=physical_high,
        _passthrough_mask=passthrough_mask,
    )


def build_act_execution_projector(
    assets: ACTPolicyAssets,
    action_domain: ACTPhysicalActionDomain,
) -> ACTExecutionProjector:
    """Build a projector preserving validated URDF and Twist semantics."""

    if not isinstance(assets, ACTPolicyAssets):
        raise TypeError("ACT action-domain projection requires ACTPolicyAssets")
    if not isinstance(action_domain, ACTPhysicalActionDomain):
        raise TypeError("ACT action-domain projection requires ACTPhysicalActionDomain")
    if assets.action_dim != action_domain.action_dim:
        raise ValueError(
            "ACT checkpoint and physical action-domain dimensions disagree: "
            f"{assets.action_dim} != {action_domain.action_dim}"
        )

    try:
        policy_parameter = next(assets.policy.parameters())
    except StopIteration as error:
        raise ValueError("ACT policy has no parameters") from error
    if not policy_parameter.is_floating_point():
        raise TypeError("ACT policy parameters must be floating point")

    projector = ACTExecutionProjector(
        action_mean=assets.action_mean,
        action_std=assets.action_std,
        normalizer_eps=assets.normalizer_eps,
        physical_low=action_domain.physical_low,
        physical_high=action_domain.physical_high,
        binary_mask=None,
        passthrough_mask=action_domain.passthrough_mask,
    )
    projector.to(device=policy_parameter.device, dtype=policy_parameter.dtype)
    if projector.action_dim != action_domain.action_dim:
        raise RuntimeError("ACT action projector dimension changed during construction")
    return projector


__all__ = [
    "ACTPhysicalActionDomain",
    "build_act_execution_projector",
    "load_act_physical_action_domain",
]
