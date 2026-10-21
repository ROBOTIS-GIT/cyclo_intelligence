"""Runtime action horizons without changing trained model/TensorRT shapes."""

from __future__ import annotations


def requested_chunk_size(value=0) -> int:
    """Zero preserves checkpoint behavior for clients predating this option."""
    try:
        parsed = int(value)
        if float(value) != parsed or not 0 <= parsed <= 65535:
            raise ValueError
    except (TypeError, ValueError, OverflowError):
        raise ValueError("action_chunk_size must be an integer from 0 to 65535") from None
    return parsed


def effective_chunk_size(requested: int, maximum: int) -> int:
    requested = requested_chunk_size(requested)
    if maximum < 1:
        raise ValueError("Checkpoint has no action horizon")
    return min(requested, maximum) if requested else maximum


def act_execution_steps(config: dict, requested=0) -> int:
    if config.get("temporal_ensemble_coeff") is not None:
        return 1
    requested = requested_chunk_size(requested)
    if requested:
        return effective_chunk_size(requested, int(config["chunk_size"]))
    return int(config.get("n_action_steps", config.get("chunk_size", 100)))
