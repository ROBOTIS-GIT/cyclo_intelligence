"""Preserve an explicit checkpoint mode, otherwise use the continuous action head."""

from .definition import AdapterDefinition


def load_policy(policy_class, config_class, model_path, device):
    policy = policy_class.from_pretrained(model_path)
    if not getattr(policy.config, "inference_action_mode", None):
        policy.config.inference_action_mode = "continuous"
    return policy.to(device).eval()


ADAPTER = AdapterDefinition(policy_loader=load_policy)
