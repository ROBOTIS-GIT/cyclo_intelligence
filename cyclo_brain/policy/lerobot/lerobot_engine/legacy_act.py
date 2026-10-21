"""Load pre-processor-migration ACT weights without losing saved statistics."""
from pathlib import Path

import torch
from safetensors.torch import load_file


def load_legacy_act(model_path, config, policy_class, device):
    """Return policy/processors for LeRobot 0.3 ACT, or None for modern weights.

    Only the normalization buffers moved in this format migration. Load all
    model tensors strictly and feed the original buffers to the official ACT
    processor factory; never estimate statistics from the current scene.
    """
    path = Path(model_path) / "model.safetensors"
    if not path.is_file():
        return None
    weights = load_file(str(path), device="cpu")
    if not any(key.startswith("normalize_inputs.") for key in weights):
        return None

    stats = {}
    consumed = set()
    for prefix, features in (
        ("normalize_inputs", config.input_features),
        ("unnormalize_outputs", config.output_features),
    ):
        for name, feature in features.items():
            mode = config.normalization_mapping[feature.type]
            if getattr(mode, "value", mode) != "MEAN_STD":
                raise ValueError(f"Legacy ACT requires MEAN_STD for {name}; got {mode}")
            values = {}
            for statistic in ("mean", "std"):
                key = f"{prefix}.buffer_{name.replace('.', '_')}.{statistic}"
                if key not in weights:
                    raise ValueError(f"Legacy ACT normalization buffer missing: {key}")
                value = weights[key]
                expected_shape = (feature.shape[0], 1, 1) if len(feature.shape) == 3 else tuple(feature.shape)
                if tuple(value.shape) != expected_shape or not torch.isfinite(value).all():
                    raise ValueError(f"Invalid legacy ACT normalization buffer: {key}")
                if statistic == "std" and torch.any(value <= 0):
                    raise ValueError(f"Non-positive legacy ACT standard deviation: {key}")
                values[statistic] = value
                consumed.add(key)
                if prefix == "unnormalize_outputs":
                    target = key.replace("unnormalize_outputs.", "normalize_targets.")
                    if target not in weights or not torch.equal(weights[target], value):
                        raise ValueError(f"Inconsistent ACT target/output statistics: {target}")
                    consumed.add(target)
            stats[name] = values

    model_weights = {key: value for key, value in weights.items() if key.startswith("model.")}
    unexpected = set(weights) - consumed - set(model_weights)
    if unexpected:
        raise ValueError(f"Unexpected legacy ACT tensors: {sorted(unexpected)}")

    # Every backbone tensor is restored below; no ImageNet download is needed.
    config.pretrained_backbone_weights = None
    config.device = str(device)
    policy = policy_class(config)
    policy.load_state_dict(model_weights, strict=True)
    policy = policy.to(device).eval()
    from lerobot.policies.act.processor_act import make_act_pre_post_processors

    preprocessor, postprocessor = make_act_pre_post_processors(config, dataset_stats=stats)
    return policy, preprocessor, postprocessor
