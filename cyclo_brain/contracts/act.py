"""Dependency-free ACT trainability and TD3 schedule rules."""
from collections.abc import Iterable

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

ACT_TD3_ACTOR_OBJECTIVES = ("td3", "td3_bc")
"""Canonical actor objectives exposed by the ACT-TD3 training contract."""


def canonicalize_act_td3_actor_objective(value: str) -> str:
    """Validate one exact, checkpoint-stable ACT-TD3 actor objective ID."""

    if not isinstance(value, str):
        raise TypeError("ACT-TD3 actor_objective must be a string")
    if value not in ACT_TD3_ACTOR_OBJECTIVES:
        raise ValueError(
            "ACT-TD3 actor_objective must be one of: "
            + ", ".join(ACT_TD3_ACTOR_OBJECTIVES)
        )
    return value

def effective_act_td3_trainable_groups(objective: str, groups: Iterable[str]) -> tuple[str, ...]:
    canonicalize_act_td3_actor_objective(objective)
    return tuple(
        group for group in canonicalize_act_trainable_groups(groups)
        if objective != "td3" or group != "cvae_encoder"
    )


def policy_update_period_for_epoch_schedule(critic_epochs: int, actor_equivalent_epochs: int) -> int:
    """Return the exact critic-to-actor update ratio, including 1:1."""
    for name, value in (
        ("critic_epochs", critic_epochs),
        ("actor_equivalent_epochs", actor_equivalent_epochs),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if critic_epochs < actor_equivalent_epochs or critic_epochs % actor_equivalent_epochs:
        raise ValueError(
            "TD3 requires critic_epochs to be an exact integer multiple of "
            "actor_equivalent_epochs; 1:1 is supported"
        )
    return critic_epochs // actor_equivalent_epochs
