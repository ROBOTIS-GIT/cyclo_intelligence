"""Trainable-parameter groups for the official LeRobot ACT policy."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from lerobot.policies.act.modeling_act import ACTPolicy


from cyclo_brain.contracts.act import (
    ACT_TRAINABLE_GROUPS,
    ACT_DETERMINISTIC_INFERENCE_GROUPS,
    canonicalize_act_trainable_groups,
)


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
