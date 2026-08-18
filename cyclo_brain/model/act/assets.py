"""Strict loading of the ACT policy and its saved processor assets."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor

from .factory import load_act_model

if TYPE_CHECKING:
    from torch import device as TorchDevice

    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.processor import PolicyProcessorPipeline


@dataclass(frozen=True)
class ACTPolicyAssets:
    """Official ACT policy plus the exact processors saved with its checkpoint.

    Action statistics are retained privately. Each public tensor access returns
    a detached clone so callers cannot mutate the validated checkpoint contract.
    """

    policy: "ACTPolicy"
    preprocessor: "PolicyProcessorPipeline[Any, Any]"
    postprocessor: "PolicyProcessorPipeline[Any, Any]"
    _action_mean: Tensor = field(repr=False)
    _action_std: Tensor = field(repr=False)
    normalizer_eps: float

    @property
    def action_mean(self) -> Tensor:
        return self._action_mean.detach().clone()

    @property
    def action_std(self) -> Tensor:
        return self._action_std.detach().clone()

    @property
    def action_dim(self) -> int:
        return int(self._action_mean.numel())


def _single_step(pipeline: Any, step_type: type, name: str) -> Any:
    matches = [step for step in pipeline.steps if isinstance(step, step_type)]
    if len(matches) != 1:
        raise ValueError(
            f"ACT saved {name} must contain exactly one {step_type.__name__}; "
            f"found {len(matches)}"
        )
    return matches[0]


def _mean_std_mode(norm_map: Any, source: str) -> None:
    from lerobot.configs import FeatureType, NormalizationMode

    mode = norm_map.get(FeatureType.ACTION)
    if mode != NormalizationMode.MEAN_STD:
        raise ValueError(
            f"ACT {source} ACTION normalization must be MEAN_STD; got {mode!r}"
        )


def _validate_step_action_feature(step: Any, action_dim: int, source: str) -> None:
    from lerobot.configs import FeatureType
    from lerobot.utils.constants import ACTION

    feature = step.features.get(ACTION)
    if (
        feature is None
        or feature.type != FeatureType.ACTION
        or tuple(feature.shape) != (action_dim,)
    ):
        raise ValueError(
            f"ACT {source} ACTION feature must have shape ({action_dim},)"
        )


def _action_stats(step: Any, action_dim: int, source: str) -> tuple[Tensor, Tensor]:
    from lerobot.utils.constants import ACTION

    stats = step.stats.get(ACTION)
    if not isinstance(stats, dict):
        raise ValueError(f"ACT saved {source} is missing ACTION statistics")

    values: list[Tensor] = []
    for stat_name in ("mean", "std"):
        if stat_name not in stats:
            raise ValueError(
                f"ACT saved {source} ACTION statistics are missing {stat_name!r}"
            )
        value = torch.as_tensor(stats[stat_name]).detach().clone()
        if (
            not value.is_floating_point()
            or value.shape != (action_dim,)
            or not bool(torch.isfinite(value).all())
        ):
            raise ValueError(
                f"ACT saved {source} ACTION {stat_name} must be a finite "
                f"floating vector with shape ({action_dim},)"
            )
        values.append(value)

    mean, std = values
    if bool((std < 0.0).any()):
        raise ValueError(f"ACT saved {source} ACTION std must be non-negative")
    return mean, std


def _validated_action_normalization(
    policy: "ACTPolicy",
    preprocessor: Any,
    postprocessor: Any,
) -> tuple[Tensor, Tensor, float]:
    from lerobot.configs import FeatureType
    from lerobot.processor import NormalizerProcessorStep, UnnormalizerProcessorStep
    from lerobot.utils.constants import ACTION

    action_feature = (policy.config.output_features or {}).get(ACTION)
    if (
        action_feature is None
        or action_feature.type != FeatureType.ACTION
        or len(action_feature.shape) != 1
        or action_feature.shape[0] < 1
    ):
        raise ValueError("ACT policy must define a one-dimensional ACTION output feature")
    action_dim = int(action_feature.shape[0])

    _mean_std_mode(policy.config.normalization_mapping, "policy config")
    normalizer = _single_step(
        preprocessor,
        NormalizerProcessorStep,
        "preprocessor",
    )
    unnormalizer = _single_step(
        postprocessor,
        UnnormalizerProcessorStep,
        "postprocessor",
    )
    _mean_std_mode(normalizer.norm_map, "preprocessor")
    _mean_std_mode(unnormalizer.norm_map, "postprocessor")
    _validate_step_action_feature(normalizer, action_dim, "preprocessor")
    _validate_step_action_feature(unnormalizer, action_dim, "postprocessor")

    pre_mean, pre_std = _action_stats(normalizer, action_dim, "preprocessor")
    post_mean, post_std = _action_stats(unnormalizer, action_dim, "postprocessor")
    if pre_mean.dtype != post_mean.dtype or not torch.equal(pre_mean, post_mean):
        raise ValueError("ACT pre/postprocessor ACTION mean statistics do not match exactly")
    if pre_std.dtype != post_std.dtype or not torch.equal(pre_std, post_std):
        raise ValueError("ACT pre/postprocessor ACTION std statistics do not match exactly")

    pre_eps = normalizer.eps
    post_eps = unnormalizer.eps
    if (
        isinstance(pre_eps, bool)
        or isinstance(post_eps, bool)
        or not isinstance(pre_eps, (int, float))
        or not isinstance(post_eps, (int, float))
        or not math.isfinite(float(pre_eps))
        or float(pre_eps) <= 0.0
        or float(pre_eps) != float(post_eps)
    ):
        raise ValueError(
            "ACT pre/postprocessor normalizer eps must be identical, finite, and positive"
        )
    return pre_mean.detach().clone(), pre_std.detach().clone(), float(pre_eps)


def load_act_policy_assets(
    checkpoint: str | Path,
    *,
    device: str | "TorchDevice" | None = None,
) -> ACTPolicyAssets:
    """Load an ACT policy and the processor pipelines stored beside it.

    Policy weights are always loaded strictly. Processor loading is deliberately
    checkpoint-only: missing processor files fail instead of silently creating
    empty-stat defaults. The preprocessor device step follows the loaded policy.
    """

    from lerobot.policies import make_pre_post_processors

    policy = load_act_model(checkpoint, device=device, strict=True)
    try:
        policy_device = next(policy.parameters()).device
    except StopIteration as error:
        raise ValueError("ACT policy has no parameters") from error

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=str(checkpoint),
        preprocessor_overrides={
            "device_processor": {"device": str(policy_device)},
        },
    )
    action_mean, action_std, normalizer_eps = _validated_action_normalization(
        policy,
        preprocessor,
        postprocessor,
    )
    return ACTPolicyAssets(
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        _action_mean=action_mean,
        _action_std=action_std,
        normalizer_eps=normalizer_eps,
    )


__all__ = ["ACTPolicyAssets", "load_act_policy_assets"]
