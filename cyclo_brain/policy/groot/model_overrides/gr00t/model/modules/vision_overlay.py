"""Strict, complete visual overlay shared by training and rendering."""

from safetensors.torch import load_file


PREFIX = "backbone.model.model.visual."


def load_visual_overlay(visual, path):
    weights = load_file(str(path))
    if not weights or any(not k.startswith(PREFIX) for k in weights):
        raise ValueError("Expected visual-only safetensors with canonical GR00T prefixes")
    state = {k[len(PREFIX) :]: v for k, v in weights.items()}
    reference = visual.state_dict()
    if set(state) != set(reference):
        raise ValueError(
            f"Overlay key mismatch: missing={set(reference) - set(state)}, extra={set(state) - set(reference)}"
        )
    for key, value in state.items():
        if value.shape != reference[key].shape:
            raise ValueError(f"Overlay shape mismatch: {key}")
    visual.load_state_dict(state, strict=True)
    return len(state)
