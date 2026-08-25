"""Strict deployment validation for Cyclo MultiTaskDiT checkpoints.

The training entrypoint saves checkpoints with LeRobot's official
``save_checkpoint`` helper.  This module validates the resulting
``pretrained_model`` directory without defining a second checkpoint format.
It intentionally keeps the contract and tensor-comparison helpers independent
of LeRobot so they can be exercised by fast host-only tests.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from .flow_sde_adapter import CYCLO_SG2_CAMERA_KEYS, MultiTaskDiTFlowAdapter


@dataclass(frozen=True)
class MultiTaskDiTDeploymentContract:
    """The initial Cyclo SG2 policy contract used by training and deployment."""

    camera_keys: tuple[str, ...] = CYCLO_SG2_CAMERA_KEYS
    state_dim: int = 22
    action_dim: int = 22
    horizon: int = 16
    n_obs_steps: int = 1
    n_action_steps: int = 16
    objective: str = "flow_matching"
    sigma_min: float = 0.0


DEFAULT_DEPLOYMENT_CONTRACT = MultiTaskDiTDeploymentContract()


@dataclass(frozen=True)
class CheckpointRoundTripResult:
    """Compact, JSON-serializable summary of a successful validation."""

    pretrained_dir: str
    state_tensor_count: int
    state_parameter_count: int
    velocity_max_abs_error: float
    preprocessor_checked: bool
    postprocessor_checked: bool

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def resolve_pretrained_model_dir(checkpoint_path: str | Path) -> Path:
    """Accept either an official step checkpoint or its deployment directory."""

    path = Path(checkpoint_path).expanduser().resolve()
    direct_files = (path / "config.json", path / "model.safetensors")
    nested = path / "pretrained_model"
    nested_files = (nested / "config.json", nested / "model.safetensors")
    if all(item.is_file() for item in direct_files):
        return path
    if all(item.is_file() for item in nested_files):
        return nested
    raise FileNotFoundError(
        f"{path} is neither a LeRobot pretrained_model directory nor an official step checkpoint"
    )


def assert_deployment_artifacts(pretrained_dir: str | Path) -> Path:
    """Require policy and processor files needed by standalone inference."""

    path = resolve_pretrained_model_dir(pretrained_dir)
    required = (
        "config.json",
        "model.safetensors",
        "policy_preprocessor.json",
        "policy_postprocessor.json",
    )
    missing = [name for name in required if not (path / name).is_file()]
    if missing:
        raise FileNotFoundError(f"MultiTaskDiT deployment checkpoint is missing {missing!r}")
    return path


def _feature_shape(feature: Any, *, name: str) -> tuple[int, ...]:
    shape = getattr(feature, "shape", None)
    if not isinstance(shape, Sequence) or isinstance(shape, (str, bytes)):
        raise ValueError(f"MultiTaskDiT config {name} must expose a shape")
    try:
        result = tuple(int(value) for value in shape)
    except (TypeError, ValueError) as error:
        raise ValueError(f"MultiTaskDiT config {name} has an invalid shape") from error
    return result


def validate_policy_contract(
    policy: nn.Module,
    *,
    contract: MultiTaskDiTDeploymentContract = DEFAULT_DEPLOYMENT_CONTRACT,
) -> dict[str, Any]:
    """Fail closed when a checkpoint cannot be executed by the SG2 adapter."""

    if not isinstance(policy, nn.Module) or not hasattr(policy, "config"):
        raise TypeError("MultiTaskDiT deployment policy must be a configured torch module")
    config = policy.config
    image_features = getattr(config, "image_features", None)
    if not isinstance(image_features, Mapping):
        raise ValueError("MultiTaskDiT config must expose ordered image_features")
    camera_keys = tuple(image_features.keys())
    if camera_keys != contract.camera_keys:
        raise ValueError(
            f"MultiTaskDiT camera order mismatch: expected {contract.camera_keys!r}, got {camera_keys!r}"
        )

    state_shape = _feature_shape(getattr(config, "robot_state_feature", None), name="state feature")
    action_shape = _feature_shape(getattr(config, "action_feature", None), name="action feature")
    if state_shape != (contract.state_dim,):
        raise ValueError(
            f"MultiTaskDiT state contract mismatch: expected {(contract.state_dim,)}, got {state_shape}"
        )
    if action_shape != (contract.action_dim,):
        raise ValueError(
            f"MultiTaskDiT action contract mismatch: expected {(contract.action_dim,)}, got {action_shape}"
        )

    expected_scalars = {
        "horizon": contract.horizon,
        "n_obs_steps": contract.n_obs_steps,
        "n_action_steps": contract.n_action_steps,
        "objective": contract.objective,
        "sigma_min": contract.sigma_min,
    }
    for name, expected in expected_scalars.items():
        actual = getattr(config, name, None)
        if actual != expected:
            raise ValueError(f"MultiTaskDiT {name} mismatch: expected {expected!r}, got {actual!r}")

    return {
        "camera_keys": list(camera_keys),
        "state_dim": state_shape[0],
        "action_dim": action_shape[0],
        **expected_scalars,
    }


def assert_exact_state_dict(
    reference: Mapping[str, Tensor],
    candidate: Mapping[str, Tensor],
) -> tuple[int, int]:
    """Assert exact key, shape, dtype, and value equality after serialization."""

    reference_keys = set(reference)
    candidate_keys = set(candidate)
    if reference_keys != candidate_keys:
        missing = sorted(reference_keys - candidate_keys)
        unexpected = sorted(candidate_keys - reference_keys)
        raise AssertionError(
            f"Checkpoint state_dict keys differ: missing={missing!r}, unexpected={unexpected!r}"
        )

    parameter_count = 0
    for name in reference:
        expected = reference[name]
        actual = candidate[name]
        if not isinstance(expected, Tensor) or not isinstance(actual, Tensor):
            raise TypeError(f"Checkpoint state_dict value {name!r} is not a tensor")
        if expected.shape != actual.shape:
            raise AssertionError(
                f"Checkpoint tensor {name!r} shape differs: {tuple(expected.shape)} != {tuple(actual.shape)}"
            )
        if expected.dtype != actual.dtype:
            raise AssertionError(
                f"Checkpoint tensor {name!r} dtype differs: {expected.dtype} != {actual.dtype}"
            )
        if not torch.equal(expected, actual):
            raise AssertionError(f"Checkpoint tensor {name!r} changed during save/load")
        parameter_count += expected.numel()
    return len(reference), parameter_count


def _assert_tensor_tree_close(
    expected: Any,
    actual: Any,
    *,
    path: str,
    rtol: float = 0.0,
    atol: float = 0.0,
) -> None:
    if isinstance(expected, Tensor):
        if not isinstance(actual, Tensor):
            raise AssertionError(f"{path} type differs: tensor != {type(actual).__name__}")
        if expected.dtype != actual.dtype:
            raise AssertionError(f"{path} tensor dtype differs: {expected.dtype} != {actual.dtype}")
        torch.testing.assert_close(expected, actual, rtol=rtol, atol=atol, msg=lambda msg: f"{path}: {msg}")
        return
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping) or set(expected) != set(actual):
            raise AssertionError(f"{path} mapping keys differ")
        for key in expected:
            _assert_tensor_tree_close(
                expected[key], actual[key], path=f"{path}.{key}", rtol=rtol, atol=atol
            )
        return
    if dataclasses.is_dataclass(expected) and not isinstance(expected, type):
        if type(actual) is not type(expected):
            raise AssertionError(f"{path} dataclass type differs")
        for field in dataclasses.fields(expected):
            _assert_tensor_tree_close(
                getattr(expected, field.name),
                getattr(actual, field.name),
                path=f"{path}.{field.name}",
                rtol=rtol,
                atol=atol,
            )
        return
    if isinstance(expected, Sequence) and not isinstance(expected, (str, bytes)):
        if not isinstance(actual, type(expected)) or len(expected) != len(actual):
            raise AssertionError(f"{path} sequence differs")
        for index, (expected_item, actual_item) in enumerate(zip(expected, actual, strict=True)):
            _assert_tensor_tree_close(
                expected_item,
                actual_item,
                path=f"{path}[{index}]",
                rtol=rtol,
                atol=atol,
            )
        return
    if expected != actual:
        raise AssertionError(f"{path} differs: {expected!r} != {actual!r}")


def compare_fixed_velocity(
    reference: MultiTaskDiTFlowAdapter,
    candidate: MultiTaskDiTFlowAdapter,
    *,
    seed: int = 17,
    batch_size: int = 2,
    rtol: float = 0.0,
    atol: float = 0.0,
) -> float:
    """Compare both action heads on one deterministic latent/progress input."""

    if reference.horizon != candidate.horizon or reference.action_dim != candidate.action_dim:
        raise AssertionError("MultiTaskDiT adapters expose different action contracts")
    if reference.conditioning_dim != candidate.conditioning_dim:
        raise AssertionError("MultiTaskDiT adapters expose different conditioning dimensions")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("Velocity comparison batch_size must be positive")

    parameter = next(reference.policy.noise_predictor.parameters())
    candidate_parameter = next(candidate.policy.noise_predictor.parameters())
    if parameter.device != candidate_parameter.device:
        raise ValueError("Reference and reloaded MultiTaskDiT policies must share one device")
    device = parameter.device
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    latent = torch.randn(
        (batch_size, reference.horizon, reference.action_dim),
        dtype=torch.float32,
        device=device,
        generator=generator,
    )
    conditioning = torch.randn(
        (batch_size, reference.conditioning_dim),
        dtype=torch.float32,
        device=device,
        generator=generator,
    )
    progress = torch.linspace(0.0, 0.75, batch_size, dtype=torch.float32, device=device)

    with torch.no_grad():
        expected = reference.velocity(latent, progress, conditioning)
        actual = candidate.velocity(latent, progress, conditioning)
    torch.testing.assert_close(expected, actual, rtol=rtol, atol=atol)
    return float((expected - actual).abs().max().item())


def _default_policy_loader(path: Path, reference_policy: nn.Module) -> nn.Module:
    policy_class = type(reference_policy)
    if not hasattr(policy_class, "from_pretrained"):
        raise TypeError("MultiTaskDiT policy class does not implement from_pretrained")
    return policy_class.from_pretrained(path, strict=True, local_files_only=True)


def _default_processor_loader(path: Path, policy: nn.Module) -> tuple[Any, Any]:
    # Kept lazy so host-only contract tests do not import the full LeRobot stack.
    from lerobot.policies.factory import make_pre_post_processors

    return make_pre_post_processors(policy.config, pretrained_path=str(path))


def _clone_tree(value: Any) -> Any:
    if isinstance(value, Tensor):
        return value.detach().clone()
    if isinstance(value, Mapping):
        return type(value)((key, _clone_tree(item)) for key, item in value.items())
    if isinstance(value, tuple):
        return tuple(_clone_tree(item) for item in value)
    if isinstance(value, list):
        return [_clone_tree(item) for item in value]
    return copy.deepcopy(value)


def validate_checkpoint_round_trip(
    reference_policy: nn.Module,
    checkpoint_path: str | Path,
    *,
    contract: MultiTaskDiTDeploymentContract = DEFAULT_DEPLOYMENT_CONTRACT,
    preprocessor: Callable[[Any], Any] | None = None,
    raw_batch: Any | None = None,
    postprocessor: Callable[[Any], Any] | None = None,
    normalized_action: Any | None = None,
    policy_loader: Callable[[Path, nn.Module], nn.Module] | None = None,
    processor_loader: Callable[[Path, nn.Module], tuple[Any, Any]] | None = None,
    adapter_factory: Callable[..., MultiTaskDiTFlowAdapter] = MultiTaskDiTFlowAdapter,
    seed: int = 17,
) -> CheckpointRoundTripResult:
    """Strictly reload and compare an official LeRobot deployment checkpoint.

    ``raw_batch`` and ``normalized_action`` are optional because training can
    validate weights before decoding another video frame.  If either is
    supplied, its matching reference processor is required and the saved
    processor is compared on the same immutable input.
    """

    if (preprocessor is None) != (raw_batch is None):
        raise ValueError("preprocessor and raw_batch must be supplied together")
    if (postprocessor is None) != (normalized_action is None):
        raise ValueError("postprocessor and normalized_action must be supplied together")

    pretrained_dir = assert_deployment_artifacts(checkpoint_path)
    reference_policy = reference_policy.module if hasattr(reference_policy, "module") else reference_policy
    validate_policy_contract(reference_policy, contract=contract)

    loader = policy_loader or _default_policy_loader
    loaded_policy = loader(pretrained_dir, reference_policy)
    loaded_policy = loaded_policy.module if hasattr(loaded_policy, "module") else loaded_policy
    validate_policy_contract(loaded_policy, contract=contract)
    state_tensor_count, state_parameter_count = assert_exact_state_dict(
        reference_policy.state_dict(), loaded_policy.state_dict()
    )

    reference_adapter = adapter_factory(
        reference_policy,
        expected_camera_keys=contract.camera_keys,
        freeze_observation_encoder=True,
    )
    loaded_adapter = adapter_factory(
        loaded_policy,
        expected_camera_keys=contract.camera_keys,
        freeze_observation_encoder=True,
    )
    if reference_policy.observation_encoder.training or loaded_policy.observation_encoder.training:
        raise AssertionError("Reloaded MultiTaskDiT observation encoder was not frozen in eval mode")
    if any(parameter.requires_grad for parameter in loaded_policy.observation_encoder.parameters()):
        raise AssertionError("Reloaded MultiTaskDiT observation encoder still requires gradients")
    velocity_error = compare_fixed_velocity(reference_adapter, loaded_adapter, seed=seed)

    preprocessor_checked = preprocessor is not None
    postprocessor_checked = postprocessor is not None
    if preprocessor_checked or postprocessor_checked:
        load_processors = processor_loader or _default_processor_loader
        loaded_preprocessor, loaded_postprocessor = load_processors(pretrained_dir, loaded_policy)
        if preprocessor_checked:
            expected = preprocessor(_clone_tree(raw_batch))
            actual = loaded_preprocessor(_clone_tree(raw_batch))
            _assert_tensor_tree_close(expected, actual, path="preprocessor")
        if postprocessor_checked:
            expected = postprocessor(_clone_tree(normalized_action))
            actual = loaded_postprocessor(_clone_tree(normalized_action))
            _assert_tensor_tree_close(expected, actual, path="postprocessor")

    return CheckpointRoundTripResult(
        pretrained_dir=str(pretrained_dir),
        state_tensor_count=state_tensor_count,
        state_parameter_count=state_parameter_count,
        velocity_max_abs_error=velocity_error,
        preprocessor_checked=preprocessor_checked,
        postprocessor_checked=postprocessor_checked,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        required=True,
        type=Path,
        help="Official LeRobot step checkpoint or its pretrained_model directory",
    )
    args = parser.parse_args()

    pretrained_dir = assert_deployment_artifacts(args.checkpoint)
    from lerobot.policies.multi_task_dit import MultiTaskDiTPolicy

    reference = MultiTaskDiTPolicy.from_pretrained(
        pretrained_dir, strict=True, local_files_only=True
    )
    result = validate_checkpoint_round_trip(reference, pretrained_dir)
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
