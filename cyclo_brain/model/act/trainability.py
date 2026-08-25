"""Trainable-parameter groups for the official LeRobot ACT policy."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from lerobot.policies.act.modeling_act import ACTPolicy


ACT_TRAINABLE_GROUPS = (
    "visual_backbone",
    "cvae_encoder",
    "transformer_encoder",
    "action_decoder",
)
"""Canonical UI, CLI, and checkpoint order for ACT parameter groups."""

ACT_DETERMINISTIC_INFERENCE_GROUPS = (
    "visual_backbone",
    "transformer_encoder",
    "action_decoder",
)
"""Groups that contribute to the deployed zero-latent inference path."""


def canonicalize_act_trainable_groups(groups: Iterable[str]) -> tuple[str, ...]:
    """Validate and return selected ACT groups in the canonical order.

    The CVAE encoder consumes target actions only while training. Allowing it
    to be the sole trainable group would produce a checkpoint whose deployed
    deterministic actor is unchanged, so that selection is rejected here.
    """

    if isinstance(groups, (str, bytes)):
        raise TypeError("ACT trainable groups must be an iterable of group names")
    try:
        requested = tuple(groups)
    except TypeError as error:
        raise TypeError(
            "ACT trainable groups must be an iterable of group names"
        ) from error
    if not requested:
        raise ValueError("ACT trainable groups cannot be empty")
    if any(not isinstance(group, str) for group in requested):
        raise TypeError("ACT trainable group names must be strings")
    if len(set(requested)) != len(requested):
        raise ValueError("ACT trainable groups cannot contain duplicates")
    unknown = sorted(set(requested).difference(ACT_TRAINABLE_GROUPS))
    if unknown:
        raise ValueError(f"Unknown ACT trainable group(s): {', '.join(unknown)}")

    canonical = tuple(
        group for group in ACT_TRAINABLE_GROUPS if group in requested
    )
    if not set(canonical).intersection(ACT_DETERMINISTIC_INFERENCE_GROUPS):
        raise ValueError(
            "ACT trainable groups must include at least one deterministic "
            "inference-path group"
        )
    return canonical


def act_parameter_group(parameter_name: str) -> str:
    """Map one official ``ACTPolicy`` parameter name to its UI group."""

    if not isinstance(parameter_name, str):
        raise TypeError("ACT parameter name must be a string")
    if parameter_name.startswith("model.backbone."):
        return "visual_backbone"
    if parameter_name.startswith("model.vae_encoder"):
        return "cvae_encoder"
    if parameter_name.startswith(
        (
            "model.decoder.",
            "model.decoder_pos_embed.",
            "model.action_head.",
        )
    ):
        return "action_decoder"
    if parameter_name.startswith("model."):
        return "transformer_encoder"
    raise ValueError(
        f"Official ACT parameter is outside the trainability contract: {parameter_name}"
    )


def classify_act_parameters(policy: Any) -> dict[str, tuple[str, ...]]:
    """Fully classify every parameter of an unwrapped official ACT policy."""

    policy = _require_act_policy(policy)
    classified: dict[str, list[str]] = {
        group: [] for group in ACT_TRAINABLE_GROUPS
    }
    parameter_count = 0
    for name, _parameter in policy.named_parameters():
        classified[act_parameter_group(name)].append(name)
        parameter_count += 1
    if parameter_count == 0:
        raise ValueError("Official ACT policy has no parameters")
    result = {group: tuple(names) for group, names in classified.items()}
    if sum(len(names) for names in result.values()) != parameter_count:
        raise RuntimeError("ACT parameter classification is incomplete")
    return result


def apply_act_trainable_groups(
    policy: Any,
    groups: Iterable[str],
) -> tuple[str, ...]:
    """Apply a validated ACT freeze mask and return its canonical group IDs."""

    policy = _require_act_policy(policy)
    canonical = canonicalize_act_trainable_groups(groups)
    classified = classify_act_parameters(policy)
    selected = set(canonical)
    trainable_names = tuple(
        name
        for group in ACT_TRAINABLE_GROUPS
        if group in selected
        for name in classified[group]
    )
    deterministic_names = tuple(
        name
        for group in ACT_DETERMINISTIC_INFERENCE_GROUPS
        if group in selected
        for name in classified[group]
    )
    if not trainable_names:
        raise ValueError("Selected ACT trainable groups contain no model parameters")
    if not deterministic_names:
        raise ValueError(
            "Selected ACT trainable groups contain no deterministic "
            "inference-path parameters"
        )

    trainable = set(trainable_names)
    for name, parameter in policy.named_parameters():
        parameter.requires_grad_(name in trainable)
    return canonical


def _require_act_policy(policy: Any) -> "ACTPolicy":
    from lerobot.policies.act.modeling_act import ACTPolicy

    if not isinstance(policy, ACTPolicy):
        raise TypeError("Expected the official LeRobot ACTPolicy")
    return policy


__all__ = [
    "ACT_DETERMINISTIC_INFERENCE_GROUPS",
    "ACT_TRAINABLE_GROUPS",
    "act_parameter_group",
    "apply_act_trainable_groups",
    "canonicalize_act_trainable_groups",
    "classify_act_parameters",
]
